# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU boundary tests; these do not claim a CUDA allocator or native DDP run."""
import importlib.util
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[2] / 'swift/megatron/trainers/model_host_cache.py'
SPEC = importlib.util.spec_from_file_location('model_host_cache_tested_source', SOURCE)
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


class Tensor:

    def __init__(self, device='cuda', count=8, element_size=2, storage_bytes=None):
        self.device = SimpleNamespace(type=device, index=0)
        self.is_cuda = device == 'cuda'
        self.shape = (count, )
        self.dtype = 'bf16'
        self.count = count
        self.width = element_size
        self.storage_bytes = count * element_size if storage_bytes is None else storage_bytes

    def numel(self):
        return self.count

    def element_size(self):
        return self.width

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: self.storage_bytes)


class DDP:

    def __init__(self, buffers, experts=()):
        self.buffers = buffers
        self.expert_parallel_buffers = list(experts)

    @property
    def module(self):
        raise AssertionError('Frozen parameters must not be visited')


def buffer():
    params, grads = Tensor(), Tensor(element_size=4)
    params.cpu_data = Tensor('cpu')
    return SimpleNamespace(param_data=params, grad_data=grads)


class HostCacheTest(TestCase):

    def invoke(self, models, synchronize=None, empty_host_cache=None, with_accelerator=True):
        torch = ModuleType('torch')
        torch.cuda = SimpleNamespace(synchronize=synchronize or Mock())
        if with_accelerator:
            torch.accelerator = SimpleNamespace()
            if empty_host_cache is not None:
                torch.accelerator.empty_host_cache = empty_host_cache
        distributed = ModuleType('megatron.core.distributed')
        distributed.DistributedDataParallel = DDP
        with patch.dict('sys.modules', {'torch': torch, 'megatron.core.distributed': distributed}):
            return CACHE.release_restored_model_host_cache(models)

    def test_sync_precedes_alias_removal_and_cache_is_not_retained_during_trim(self):
        dense, expert = buffer(), buffer()
        weak = weakref.ref(expert.param_data.cpu_data)
        params = dense.param_data
        grads = dense.grad_data
        optimizer = {'unrelated_owned_tensor': object()}
        original_optimizer = dict(optimizer)
        events = []

        def synchronize(device):
            self.assertEqual(device.type, 'cuda')
            self.assertTrue(hasattr(dense.param_data, 'cpu_data'))
            self.assertTrue(hasattr(expert.param_data, 'cpu_data'))
            events.append('sync')

        def trim():
            self.assertEqual(events, ['sync'])
            self.assertFalse(hasattr(dense.param_data, 'cpu_data'))
            self.assertFalse(hasattr(expert.param_data, 'cpu_data'))
            self.assertIsNone(weak())
            events.append('trim')

        self.assertEqual(self.invoke([DDP([dense], [expert])], synchronize, trim), 32)
        self.assertEqual(events, ['sync', 'trim'])
        self.assertIs(dense.param_data, params)
        self.assertIs(dense.grad_data, grads)
        self.assertEqual((params.storage_bytes, grads.storage_bytes), (16, 32))
        self.assertEqual(optimizer, original_optimizer)

    def test_partial_restore_or_wrong_cache_fails_before_any_mutation(self):
        for field, value in (('param_storage', 0), ('param_storage', 8), ('grad_storage', 0), ('grad_storage', 16),
                             ('grad_device', 'cpu'), ('cache_device', 'cuda'), ('cache_shape', (1, )), ('cache_dtype',
                                                                                                        'fp32')):
            first, bad = buffer(), buffer()
            if field == 'param_storage':
                bad.param_data.storage_bytes = value
            elif field == 'grad_storage':
                bad.grad_data.storage_bytes = value
            elif field == 'grad_device':
                bad.grad_data = Tensor(value, element_size=4)
            else:
                setattr(bad.param_data.cpu_data, field.removeprefix('cache_'),
                        SimpleNamespace(type=value) if field == 'cache_device' else value)
            sync = Mock()
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.invoke([DDP([first, bad])], sync)
            sync.assert_not_called()
            self.assertTrue(hasattr(first.param_data, 'cpu_data'))
            self.assertTrue(hasattr(bad.param_data, 'cpu_data'))

    def test_synchronization_failure_preserves_all_copy_sources(self):
        value = buffer()
        with self.assertRaises(RuntimeError):
            self.invoke([DDP([value])], Mock(side_effect=RuntimeError('synthetic copy failure')))
        self.assertTrue(hasattr(value.param_data, 'cpu_data'))

    def test_plain_reference_models_rejected_before_detaching_ddp(self):
        value = buffer()
        with self.assertRaisesRegex(ValueError, 'native DDP'):
            self.invoke([DDP([value]), object()])
        self.assertTrue(hasattr(value.param_data, 'cpu_data'))

    def test_duplicate_wrappers_do_not_count_or_remove_cache_twice(self):
        value = buffer()
        ddp = DDP([value])
        self.assertEqual(self.invoke([ddp, ddp]), 16)

    def test_optional_inactive_cache_api_and_no_cache_are_safe(self):
        for with_accelerator in (True, False):
            with self.subTest(with_accelerator=with_accelerator):
                self.assertEqual(self.invoke([DDP([buffer()])], with_accelerator=with_accelerator), 16)
        value = buffer()
        del value.param_data.cpu_data
        sync, trim = Mock(), Mock()
        self.assertEqual(self.invoke([DDP([value])], sync, trim), 0)
        sync.assert_not_called()
        trim.assert_not_called()

    def test_external_cpu_owner_is_preserved(self):
        value = buffer()
        external = value.param_data.cpu_data
        self.assertEqual(self.invoke([DDP([value])]), 16)
        self.assertEqual((external.device.type, external.numel()), ('cpu', 8))


if __name__ == '__main__':
    main()
