# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real CPU Tensor/file checks plus the actual trainer methods with native stubs.

No CUDA, distributed initialization, weights downloads or scheduler operations.
Only the filesystem classification is mocked for temporary-directory I/O; its
allowlist is tested separately. These tests do not establish GPU pressure fit.
"""
import ast
import contextlib
import importlib.util
import os
import stat
import sys
import tempfile
import torch
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'swift/megatron/trainers/reference_cache.py'
SPEC = importlib.util.spec_from_file_location('reference_cache_test_module', SOURCE)
cache_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache_module)


def model(dtype=torch.float32):
    result = torch.nn.Module()
    result.register_parameter('weight', torch.nn.Parameter(torch.arange(12, dtype=dtype).reshape(3, 4), False))
    result.eval()
    return result


def args(directory=None, **changes):
    value = dict(
        reference_model_cache_dir=directory,
        reference_model_cache_allow_nfs=False,
        release_model_cpu_cache=False,
        rlhf_type='grpo',
        tuner_type='full',
        offload_model=True,
        use_vllm=True,
        vllm_mode='colocate',
        use_cpu_initialization=False,
        beta=0.01,
        ref_model=None,
        mcore_ref_model=None,
        ref_adapters=None,
        mcore_ref_adapter=None,
        model='public-offline-fixture',
        use_hf=False,
        hub_token=None)
    value.update(changes)
    return types.SimpleNamespace(**value)


class CacheFiles(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.caches = []
        self.fs = patch.object(cache_module, 'disk_filesystem', return_value='ext4')
        self.fs.start()

    def tearDown(self):
        for cache in self.caches:
            # Drop real mapped Tensor views only after detaching test parameters.
            for _, parameter in cache.parameters:
                parameter.data = parameter.detach().clone()
            cache.views.clear()
            if cache.map is not None:
                cache.map.close()
            for name in ('fd', 'dir_fd', 'parent_fd'):
                value = getattr(cache, name, None)
                if value is not None:
                    os.close(value)
        self.fs.stop()
        self.tmp.cleanup()

    def create(self, reference=None, **changes):
        options = dict(rank=0, world_size=1, local_world_size=1, chunk_bytes=4096)
        options.update(changes)
        cache = cache_module.ReferenceFileCache([reference or model()], self.root, **options)
        self.caches.append(cache)
        return cache

    def test_disk_capacity_uses_local_ranks_and_nfs_uses_global_ranks(self):
        # One tiny parameter occupies one mmap page. Capacity includes the
        # existing per-rank reserve without allocating that reserve in this test.
        per_rank = cache_module.mmap.PAGESIZE + cache_module.CHUNK_BYTES
        for filesystem, replicas in (('ext4', 4), ('xfs', 4), ('btrfs', 4), ('nfs', 56), ('nfs4', 56)):
            for enough in (False, True):
                free = replicas * per_rank - (0 if enough else 1)
                space = types.SimpleNamespace(f_bavail=free, f_frsize=1)
                with self.subTest(filesystem=filesystem, enough=enough), \
                        patch.object(cache_module, 'disk_filesystem', return_value=filesystem), \
                        patch.object(cache_module.os, 'fstatvfs', return_value=space):
                    cache = self.create(world_size=56, local_world_size=4, allow_nfs=True)
                    self.assertEqual(cache.capacity_replicas, replicas)
                    original_pointer = cache.parameters[0][1].data_ptr()
                    if enough:
                        cache.prepare()
                        self.assertTrue(cache.ready)
                    else:
                        with self.assertRaisesRegex(cache_module.CacheError, 'scratch_insufficient_space'):
                            cache.prepare()
                        self.assertEqual(cache.parameters[0][1].data_ptr(), original_pointer)
                        self.assertFalse((self.root / cache.cache_name).exists())
                        self.assertTrue(all(getattr(cache, name) is None for name in ('fd', 'dir_fd', 'parent_fd')))

    def test_nfs_rejects_capacity_sufficient_only_for_local_ranks(self):
        space = types.SimpleNamespace(f_bavail=4 * (cache_module.mmap.PAGESIZE + cache_module.CHUNK_BYTES), f_frsize=1)
        with patch.object(cache_module, 'disk_filesystem', return_value='nfs4'), \
                patch.object(cache_module.os, 'fstatvfs', return_value=space):
            cache = self.create(world_size=56, local_world_size=4, allow_nfs=True)
            with self.assertRaisesRegex(cache_module.CacheError, 'scratch_insufficient_space'):
                cache.prepare()

    def test_invalid_local_size_rejected_before_opening_directory(self):
        for value in (True, False, 0, -1, 57, 3, '4', None):
            with self.subTest(value=value), patch.object(cache_module.os, 'open') as open_file, \
                    self.assertRaisesRegex(cache_module.CacheError, 'local_world_size_invalid'):
                self.create(world_size=56, local_world_size=value)
            open_file.assert_not_called()

    def test_real_cpu_roundtrip_private_files_and_repeat_reuse(self):
        cache = self.create(model(torch.bfloat16))
        parameter = cache.parameters[0][1]
        original = parameter.detach().clone()
        identity = id(parameter)
        cache.prepare()
        path = self.root / cache.cache_name
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((path / 'weights.bin').stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE((path / 'complete.json').stat().st_mode), 0o600)
        for _ in range(3):
            cache.offload()
            self.assertFalse(parameter.is_pinned())
            self.assertEqual(id(parameter), identity)
            torch.testing.assert_close(parameter, original, rtol=0, atol=0)
            parameter.data = parameter.detach().clone()  # Native H2D stand-in, CPU-only.
            with self.assertRaisesRegex(cache_module.CacheError, 'restore_copy_completion_required'):
                cache.after_restore()
            cache.after_restore(copies_completed=True)
        self.assertTrue(cache.last_file_check['content_equal'])
        self.assertFalse(cache.last_restore_observation['content_verified_at_this_boundary'])

    def test_full_digest_rejects_modified_bytes_before_parameter_rebind(self):
        cache = self.create()
        cache.prepare()
        parameter = cache.parameters[0][1]
        pointer = parameter.data_ptr()
        original = os.pread(cache.fd, 1, 0)
        os.pwrite(cache.fd, bytes([original[0] ^ 1]), 0)
        os.fsync(cache.fd)
        with self.assertRaisesRegex(cache_module.CacheError, 'cache_content_changed'):
            cache.offload()
        self.assertEqual(parameter.data_ptr(), pointer)

    def test_timestamp_only_change_requires_complete_content_match(self):
        cache = self.create()
        cache.prepare()
        path = self.root / cache.cache_name / 'weights.bin'
        info = path.stat()
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
        cache.offload()
        self.assertTrue(cache.last_file_check['metadata_changed']['mtime_ns'])
        self.assertTrue(cache.last_file_check['content_equal'])
        self.assertEqual(cache.last_file_check['bytes_read'], cache.expected_file_bytes)

    def test_parameter_and_view_versions_rejected(self):
        cache = self.create()
        cache.prepare()
        with torch.no_grad():
            cache.parameters[0][1].add_(1)
        with self.assertRaisesRegex(cache_module.CacheError, 'reference_identity_or_version_changed'):
            cache.offload()
        other = self.create()
        other.prepare()
        other.views[0].add_(1)
        with self.assertRaisesRegex(cache_module.CacheError, 'cache_view_changed'):
            other.offload()

    def test_inode_replacement_and_permissions_rejected(self):
        cache = self.create()
        cache.prepare()
        path = self.root / cache.cache_name / 'weights.bin'
        path.rename(path.with_suffix('.old'))
        path.write_bytes(b'\0' * cache.expected_file_bytes)
        os.chmod(path, 0o600)
        with self.assertRaisesRegex(cache_module.CacheError, 'cache_file_changed'):
            cache.offload()
        other = self.create()
        other.prepare()
        os.chmod(self.root / other.cache_name / 'weights.bin', 0o640)
        with self.assertRaisesRegex(cache_module.CacheError, 'cache_file_changed'):
            other.offload()

    def test_unowned_layouts_and_storage_aliases_fail_before_files(self):
        for kind in ('trainable', 'noncontiguous', 'partial_storage', 'alias'):
            reference = model()
            if kind == 'trainable':
                reference.weight.requires_grad_(True)
            elif kind == 'noncontiguous':
                reference.weight = torch.nn.Parameter(torch.arange(12.).reshape(3, 4).T, False)
            elif kind == 'partial_storage':
                reference.weight = torch.nn.Parameter(torch.arange(24.)[:12].reshape(3, 4), False)
            else:
                reference.register_parameter('alias', torch.nn.Parameter(reference.weight.detach(), False))
            with self.subTest(kind=kind), self.assertRaises(cache_module.CacheError):
                self.create(reference)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_actor_and_teacher_aliases_rejected(self):
        reference = model()
        cache_module.reject_reference_aliases([reference], [model(), model()])
        with self.assertRaisesRegex(cache_module.CacheError, 'model_alias'):
            cache_module.reject_reference_aliases([reference], [reference])
        other = model()
        other.weight = torch.nn.Parameter(reference.weight.detach(), False)
        with self.assertRaisesRegex(cache_module.CacheError, 'storage_alias'):
            cache_module.reject_reference_aliases([reference], [other])

    def test_failed_prepare_releases_descriptors_and_does_not_rebind_parameters(self):
        cache = self.create()
        parameter = cache.parameters[0][1]
        pointer = parameter.data_ptr()
        with patch.object(cache, '_file_digest', side_effect=cache_module.CacheError('injected_io_failure')):
            with self.assertRaisesRegex(cache_module.CacheError, 'injected_io_failure'):
                cache.prepare()
        self.assertEqual(parameter.data_ptr(), pointer)
        self.assertIsNone(cache.map)
        self.assertEqual(cache.views, [])
        self.assertTrue(all(getattr(cache, name) is None for name in ('fd', 'dir_fd', 'parent_fd')))
        self.assertFalse(cache.ready)

    def test_repeat_prepare_does_not_close_a_live_mapping(self):
        cache = self.create()
        cache.prepare()
        cache.offload()
        expected = cache.parameters[0][1].detach().clone()
        pointer = cache.parameters[0][1].data_ptr()
        with self.assertRaisesRegex(cache_module.CacheError, 'cache_prepare_once'):
            cache.prepare()
        self.assertTrue(cache.ready)
        self.assertFalse(cache.map.closed)
        self.assertEqual(cache.parameters[0][1].data_ptr(), pointer)
        torch.testing.assert_close(cache.parameters[0][1], expected, rtol=0, atol=0)
        self.assertTrue(all(getattr(cache, name) is not None for name in ('fd', 'dir_fd', 'parent_fd')))

    def test_mid_copy_failure_closes_resources_preserves_original_parameters(self):
        reference = model()
        reference.weight = torch.nn.Parameter(torch.arange(3000.), False)
        expected = reference.weight.detach().clone()
        cache = self.create(reference)
        original = torch.Tensor.copy_
        calls = []

        def copy_or_fail(destination, source, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise cache_module.CacheError('copy_failed')
            return original(destination, source, **kwargs)

        with patch.object(torch.Tensor, 'copy_', copy_or_fail):
            with self.assertRaisesRegex(cache_module.CacheError, 'copy_failed'):
                cache.prepare()
        torch.testing.assert_close(reference.weight, expected, rtol=0, atol=0)
        self.assertIsNone(cache.map)
        self.assertTrue(all(getattr(cache, name) is None for name in ('fd', 'dir_fd', 'parent_fd')))

    def test_failed_constructor_closes_owned_directory_descriptor(self):
        with patch.object(cache_module, 'disk_filesystem', side_effect=cache_module.CacheError('filesystem_denied')), \
                patch.object(cache_module.os, 'close', wraps=os.close) as close:
            with self.assertRaisesRegex(cache_module.CacheError, 'filesystem_denied'):
                self.create()
        self.assertEqual(close.call_count, 1)

    def test_private_path_and_no_old_cache_adoption(self):
        first = self.create()
        first.prepare()
        second = self.create()
        second.prepare()
        self.assertNotEqual(first.cache_name, second.cache_name)
        symlink = self.root / 'alias'
        symlink.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(cache_module.CacheError):
            cache_module.ReferenceFileCache([model()], symlink, rank=0, world_size=1, local_world_size=1)
        os.chmod(self.root, 0o777)
        with self.assertRaisesRegex(cache_module.CacheError, 'scratch_mutable_by_others'):
            self.create()
        os.chmod(self.root, 0o700)


class ScopeAndNativeBoundary(unittest.TestCase):

    def test_launch_local_size_requires_explicit_consistent_metadata(self):
        for settings in ({'LOCAL_WORLD_SIZE': '4'}, {'LOCAL_SIZE': '4'}, {'LOCAL_WORLD_SIZE': '4', 'LOCAL_SIZE': '4'}):
            with self.subTest(settings=settings), patch.dict(os.environ, settings, clear=True):
                self.assertEqual(cache_module.reference_cache_local_world_size(args(local_world_size=4), 56), 4)
        for settings, configured, error in (({}, 1, 'local_world_size_missing'), ({
                'LOCAL_WORLD_SIZE': '4',
                'LOCAL_SIZE': '8'
        }, 4, 'local_world_size_conflict'), ({
                'LOCAL_WORLD_SIZE': '4'
        }, 8, 'local_world_size_args_mismatch'), ({
                'LOCAL_WORLD_SIZE': '1'
        }, True, 'local_world_size_args_mismatch'), ({
                'LOCAL_WORLD_SIZE': '4'
        }, None, 'local_world_size_args_mismatch')):
            with self.subTest(settings=settings, configured=configured), \
                    patch.dict(os.environ, settings, clear=True), \
                    self.assertRaisesRegex(cache_module.CacheError, error):
                cache_module.reference_cache_local_world_size(args(local_world_size=configured), 56)
        for value in ('', '0', '-1', 'true', '4.0', ' 4', '3', '57', '1000000000'):
            with self.subTest(value=value), patch.dict(os.environ, {'LOCAL_WORLD_SIZE': value}, clear=True), \
                    self.assertRaisesRegex(cache_module.CacheError, 'local_world_size_invalid'):
                cache_module.reference_cache_local_world_size(args(local_world_size=4), 56)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(cache_module.reference_cache_local_world_size(args(), 1), 1)

    def test_disabled_cache_does_not_read_local_launch_settings(self):
        klass, events = self.native_class()
        trainer = object.__new__(klass)
        trainer.args = args()
        trainer.ref_models = [model()]
        with patch.object(cache_module, 'reference_cache_local_world_size', side_effect=AssertionError('unexpected')):
            trainer._offload_reference_model()
            trainer._restore_reference_model()
        self.assertEqual([event for event, _ in events], ['offload', 'restore'])

    def test_filesystem_allowlist_and_nfs_opt_in(self):
        for kind in ('tmpfs', 'overlay', 'nfs4', 'ext4'):
            mount = f'1 0 0:1 / / rw - {kind} device rw\n'
            if kind == 'ext4':
                self.assertEqual(cache_module.disk_filesystem(Path('/scratch'), mount), 'ext4')
            else:
                with self.assertRaises(cache_module.CacheError):
                    cache_module.disk_filesystem(Path('/scratch'), mount)
        self.assertEqual(
            cache_module.disk_filesystem(Path('/scratch'), '1 0 0:1 / / rw - nfs4 d rw', allow_nfs=True), 'nfs4')

    def test_scope_defaults_and_unsupported_modes(self):
        self.assertFalse(cache_module.validate_reference_cache_args(args()))
        self.assertTrue(cache_module.validate_reference_cache_args(args('/owned')))
        for changes in ({
                'rlhf_type': 'dpo'
        }, {
                'tuner_type': 'lora'
        }, {
                'offload_model': False
        }, {
                'vllm_mode': 'server'
        }, {
                'use_cpu_initialization': True
        }, {
                'beta': float('nan')
        }, {
                'beta': 0,
                'ref_model': 'explicit'
        }, {
                'reference_model_cache_allow_nfs': 1
        }):
            with self.subTest(changes=changes), self.assertRaises(cache_module.CacheError):
                cache_module.validate_reference_cache_args(args('/owned', **changes))
        with self.assertRaises(cache_module.CacheError):
            cache_module.validate_reference_cache_args(args(reference_model_cache_allow_nfs=True))
        self.assertFalse(cache_module.validate_reference_cache_args(args(release_model_cpu_cache=True)))
        with self.assertRaises(cache_module.CacheError):
            cache_module.validate_reference_cache_args(args(release_model_cpu_cache=True, rlhf_type='dpo'))

    def test_native_argument_validator_runs_before_expensive_super(self):
        from typing import Literal, Optional
        events = []

        class Base:

            def __post_init__(self):
                events.append('super')
                if self.beta is None:
                    self.beta = .04

        path = ROOT / 'swift/megatron/arguments/rlhf_args.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        node.decorator_list = []  # Fields are assigned explicitly, no model/distributed argument initialization.
        env = dict(MegatronSftArguments=Base, Literal=Literal, Optional=Optional)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        cls = env['MegatronRLHFArguments']
        with patch.dict(sys.modules, {'swift.megatron.trainers.reference_cache': cache_module}):
            obj = object.__new__(cls)
            obj.__dict__.update(vars(args('/owned', beta=None)))
            obj.__post_init__()
            self.assertEqual(obj.beta, .04)
            self.assertEqual(events, ['super'])
            events.clear()
            obj.rlhf_type = 'dpo'
            with self.assertRaises(cache_module.CacheError):
                obj.__post_init__()
            self.assertEqual(events, [])
        with self.assertRaisesRegex(cache_module.CacheError, 'mutable_reference_unsupported'):
            cache_module.validate_reference_cache_args(args('/owned', sync_ref_model=True))

    def native_class(self):
        # Execute the actual checked-in class, not a duplicated implementation.
        path = ROOT / 'swift/megatron/trainers/rlhf_mixin.py'
        node = next(
            n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == 'MegatronRLHFTrainer')
        events = []

        class Base:

            def prepare_model(self):
                self.unwrapped_models = [model()]

        def restore(models):
            events.append(('restore', models))
            for item in models:
                for parameter in item.parameters():
                    parameter.data = parameter.detach().clone()

        env = dict(
            __package__='swift.megatron.trainers',
            BaseMegatronTrainer=Base,
            contextmanager=contextlib.contextmanager,
            get_mcore_model=lambda *_: [model()],
            get_current_device=lambda: 'cpu',
            safe_snapshot_download=lambda *_a, **_kw: 'offline',
            offload_megatron_model_to_cpu=lambda models: events.append(('offload', models)),
            load_megatron_model_to_gpu=restore)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        return env['MegatronRLHFTrainer'], events

    def test_actual_prepare_model_beta_zero_and_positive_missing_refs(self):
        klass, events = self.native_class()
        with patch.dict(sys.modules, {'swift.megatron.trainers.reference_cache': cache_module}):
            for directory, beta, expected in ((None, 0., 1), ('/not-opened', 0., 0), ('/not-opened', .01, 1)):
                trainer = object.__new__(klass)
                trainer.args = args(directory, beta=beta)
                trainer.template = types.SimpleNamespace(config=None)
                trainer.bridge = types.SimpleNamespace(load_weights=Mock())
                trainer.prepare_model()
                self.assertEqual(len(trainer.ref_models), expected)
                self.assertFalse(hasattr(trainer, '_reference_file_cache'))
            trainer.ref_models = []
            with self.assertRaisesRegex(cache_module.CacheError, 'missing_reference_models'):
                trainer._offload_reference_model()
        self.assertEqual(events, [])

    def test_native_hooks_are_lazy_repeatable_and_reference_only(self):
        klass, events = self.native_class()
        trainer = object.__new__(klass)
        modules = {'swift.megatron.trainers.reference_cache': cache_module}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), \
                patch.object(cache_module, 'disk_filesystem', return_value='ext4'), \
                patch.dict(os.environ, {'LOCAL_WORLD_SIZE': '4'}, clear=True), \
                patch.object(torch.distributed, 'is_initialized', return_value=True), \
                patch.object(torch.distributed, 'get_world_size', return_value=56), \
                patch.object(torch.distributed, 'get_rank', return_value=0):
            trainer.args = args(tmp, local_world_size=4)
            trainer.ref_models = [model()]
            trainer.unwrapped_models = [model()]
            trainer.teacher_models = [model()]
            # Simulates native checkpoint loading after prepare_model, before
            # first offload. Cache version capture must happen after this write.
            with torch.no_grad():
                trainer.ref_models[0].weight.add_(3)
            before = trainer.ref_models[0].weight.detach().clone()
            for _ in range(2):
                trainer._offload_reference_model()
                trainer._restore_reference_model()
                torch.testing.assert_close(trainer.ref_models[0].weight, before, rtol=0, atol=0)
            self.assertEqual([name for name, _ in events], ['offload', 'restore'] * 2)
            self.assertTrue(all(models is trainer.ref_models for _, models in events))
            trainer.teacher_models[0].weight = torch.nn.Parameter(trainer.ref_models[0].weight.detach(), False)
            with self.assertRaisesRegex(cache_module.CacheError, 'storage_alias'):
                trainer._offload_reference_model()
            self.assertEqual(len(events), 4)
            cache = trainer._reference_file_cache
            self.assertEqual(cache.identity['world_size'], 56)
            self.assertEqual(cache.identity['local_world_size'], 4)
            self.assertEqual(cache.capacity_replicas, 4)
            cache.views.clear()
            cache.map.close()
            for fd in (cache.fd, cache.dir_fd, cache.parent_fd):
                os.close(fd)


if __name__ == '__main__':
    unittest.main()
