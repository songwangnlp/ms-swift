# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in disk backing for an immutable, frozen reference model.

Only owned, private, newly created files are used; no previous cache is adopted.
Full bounded-buffer checksums validate every reuse. Model values, parameter
names and credentials are never logged. Reclaim advice is not a RAM/HBM metric.
The caller retains native model restore and must drain H2D before after_restore.
"""
from __future__ import annotations

import hashlib
import json
import logging
import mmap
import os
import re
import secrets
import stat
import time
from pathlib import Path

logger = logging.getLogger(__name__)
DISK_FILESYSTEMS = frozenset({'ext4', 'xfs', 'btrfs'})
CHUNK_BYTES = 256 * 1024**2
MAX_PARAMETERS = 100_000
VERIFY_BYTES = 4 * 1024**2


class CacheError(RuntimeError):
    """Messages are fixed enums and never include paths or model content."""


def require(ok, code):
    if not ok:
        raise CacheError(code)


def _unescape_mount(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)


def disk_filesystem(path, mountinfo, *, allow_nfs=False):
    """Require the longest containing mount to be a known disk filesystem."""
    path = Path(path)
    matches = []
    for line in mountinfo.splitlines():
        left, sep, right = line.partition(' - ')
        if not sep:
            continue
        fields, more = left.split(), right.split()
        if len(fields) < 5 or len(more) < 1:
            continue
        mount = Path(_unescape_mount(fields[4]))
        if path == mount or mount in path.parents:
            matches.append((len(mount.parts), more[0]))
    require(bool(matches), 'scratch_mount_unknown')
    kind = max(matches)[1]
    require(kind in DISK_FILESYSTEMS or (allow_nfs is True and kind in ('nfs', 'nfs4')), 'scratch_not_verified_disk')
    return kind


def _owned_directory(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path, 'scratch_not_canonical')
    for item in (path, *path.parents):
        require(not item.is_symlink(), 'scratch_symlink')
    info = path.stat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid(), 'scratch_owner')
    require(info.st_mode & 0o022 == 0, 'scratch_mutable_by_others')
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    actual = os.fstat(fd)
    if (actual.st_dev, actual.st_ino) != (info.st_dev, info.st_ino):
        os.close(fd)
        raise CacheError('scratch_changed')
    return path, fd


def reference_parameters(models):
    import torch
    require(type(models) in (list, tuple) and 0 < len(models) <= 8, 'reference_models_invalid')
    params, seen, storage_owners = [], set(), set()
    for model in models:
        require(isinstance(model, torch.nn.Module) and model.training is False, 'reference_not_eval')
        for name, param in model.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            require(len(seen) <= MAX_PARAMETERS, 'reference_too_many_parameters')
            require(
                type(param) is torch.nn.Parameter and not param.requires_grad, 'reference_not_plain_frozen_parameter')
            require(
                param.layout == torch.strided and not param.is_quantized and param.is_contiguous()
                and param.storage_offset() == 0, 'reference_layout_unsupported')
            storage = param.untyped_storage()
            size = param.numel() * param.element_size()
            require(storage.nbytes() == size, 'reference_partial_storage')
            key = (param.device.type, param.device.index, storage._cdata)
            require(size == 0 or key not in storage_owners, 'reference_distinct_storage_alias')
            storage_owners.add(key)
            del storage
            require(param.device.type in ('cpu', 'cuda'), 'reference_device_unsupported')
            params.append((hashlib.sha256(name.encode()).hexdigest(), param))
    require(bool(params), 'reference_empty')
    return params


class ReferenceFileCache:
    """One process/rank cache. Mapped pages remain clean and reclaimable.

    No cache is adopted from disk. The original GPU values are copied only once;
    subsequent offloads reuse the fixed reference. Only exact complete contiguous
    storages are admitted. Unsupported aliases fail before any parameter changes.
    """

    def __init__(self,
                 models,
                 scratch,
                 *,
                 rank,
                 world_size,
                 local_world_size,
                 chunk_bytes=CHUNK_BYTES,
                 allow_nfs=False):
        require(type(rank) is int and type(world_size) is int and 0 <= rank < world_size, 'rank_invalid')
        _validate_local_size(world_size, local_world_size)
        require(type(allow_nfs) is bool, 'nfs_option_invalid')
        run_id = secrets.token_hex(16)
        require(
            type(chunk_bytes) is int and 4096 <= chunk_bytes <= CHUNK_BYTES and chunk_bytes % 4096 == 0,
            'chunk_invalid')
        self.models = models
        self.parameters = reference_parameters(models)
        self.specs = [(id(p), p.dtype, tuple(p.shape), tuple(p.stride()), p._version, str(p.device))
                      for _, p in self.parameters]
        self.inventory = [
            dict(
                index=index,
                name_sha256=name,
                shape=list(p.shape),
                stride=list(p.stride()),
                dtype=str(p.dtype),
                numel=p.numel(),
                logical_storage_bytes=p.numel() * p.element_size()) for index, (name, p) in enumerate(self.parameters)
        ]
        self.inventory_sha256 = hashlib.sha256(
            json.dumps(self.inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.identity = dict(
            schema='reference-file-cache-v1',
            cache_id=run_id,
            rank=rank,
            world_size=world_size,
            local_world_size=local_world_size,
            pid=os.getpid())
        self.chunk_bytes = chunk_bytes
        self.ready = False
        self.views = []
        self.map = None
        self.fd = None
        self.dir_fd = None
        self.scratch, self.parent_fd = _owned_directory(scratch)
        try:
            with Path('/proc/self/mountinfo').open(encoding='utf-8') as stream:
                mountinfo = stream.read(4_000_001)
            require(len(mountinfo) <= 4_000_000, 'mountinfo_oversized')
            self.filesystem = disk_filesystem(self.scratch, mountinfo, allow_nfs=allow_nfs)
            self.capacity_replicas = world_size if self.filesystem in ('nfs', 'nfs4') else local_world_size
        except BaseException:
            os.close(self.parent_fd)
            self.parent_fd = None
            raise
        self.cache_name = f'reference-{run_id}-rank{rank:03d}-pid{os.getpid()}'
        self.last_file_check = None
        self.last_restore_observation = None

    @staticmethod
    def _numeric_stat(info):
        return dict(
            device=info.st_dev,
            inode=info.st_ino,
            bytes=info.st_size,
            mtime_ns=info.st_mtime_ns,
            ctime_ns=info.st_ctime_ns,
            mode=stat.S_IMODE(info.st_mode),
            links=info.st_nlink,
            owner_matches=info.st_uid == os.geteuid())

    def _file_stat(self):
        """Bind both retained descriptors and current owned pathname; no reopen."""
        info = os.fstat(self.fd)
        named = os.stat('weights.bin', dir_fd=self.dir_fd, follow_symlinks=False)
        parent = os.fstat(self.parent_fd)
        parent_named = os.stat(self.scratch, follow_symlinks=False)
        directory = os.fstat(self.dir_fd)
        directory_named = os.stat(self.cache_name, dir_fd=self.parent_fd, follow_symlinks=False)
        valid = (
            stat.S_ISREG(info.st_mode) and stat.S_ISREG(named.st_mode) and info.st_uid == named.st_uid == os.geteuid()
            and stat.S_IMODE(info.st_mode) == stat.S_IMODE(named.st_mode) == 0o600
            and info.st_nlink == named.st_nlink == 1 and info.st_size == self.expected_file_bytes
            and (info.st_dev, info.st_ino, info.st_size) == (named.st_dev, named.st_ino, named.st_size)
            and stat.S_ISDIR(parent_named.st_mode) and stat.S_ISDIR(directory_named.st_mode)
            and (parent.st_dev, parent.st_ino) == (parent_named.st_dev, parent_named.st_ino)
            and (directory.st_dev, directory.st_ino) == (directory_named.st_dev, directory_named.st_ino)
            and parent.st_uid == directory.st_uid == os.geteuid() and parent.st_mode & 0o022 == 0
            and stat.S_IMODE(directory.st_mode) == 0o700)
        current = self._numeric_stat(info)
        expected = getattr(self, 'file_identity', None)
        if expected is not None:
            valid = valid and all(current[k] == expected[k]
                                  for k in ('device', 'inode', 'bytes', 'mode', 'links', 'owner_matches'))
        if not valid:
            logger.error('Reference cache file identity changed')
            raise CacheError('cache_file_changed')
        return current

    def _file_digest(self, before):
        """Hash all file bytes without tensor conversion or whole-file copies."""
        digest = hashlib.sha256()
        for offset in range(0, before['bytes'], VERIFY_BYTES):
            width = min(VERIFY_BYTES, before['bytes'] - offset)
            block = os.pread(self.fd, width, offset)
            require(len(block) == width, 'cache_read_incomplete')
            digest.update(block)
            del block
            self._advise_file_reclaim(offset, width)
        after = self._file_stat()
        if after != before:
            logger.error('Reference cache file changed during verification')
            raise CacheError('cache_file_changed_during_read')
        return digest.hexdigest(), after

    def _verify_contents(self, phase):
        require(phase == 'offload', 'cache_verify_phase_invalid')
        require(tuple(view._version for view in self.views) == self.view_versions, 'cache_view_changed')
        started = time.monotonic()
        before = self._file_stat()
        digest, after = self._file_digest(before)
        changed = {k: self.file_identity[k] != after[k] for k in ('mtime_ns', 'ctime_ns')}
        event = dict(
            rank=self.identity['rank'],
            pid=os.getpid(),
            phase=phase,
            before=self.file_identity,
            after=after,
            metadata_changed=changed,
            content_equal=digest == self.content_sha256,
            bytes_read=before['bytes'],
            maximum_read_buffer_bytes=VERIFY_BYTES,
            seconds=time.monotonic() - started)
        logger.debug('Reference cache verification: %s', json.dumps(event, sort_keys=True))
        require(digest == self.content_sha256, 'cache_content_changed')
        self.last_file_check = event
        # Refresh timestamps only after complete byte parity and stable metadata.
        self.file_identity = after
        return event

    def _validate(self, *, placement):
        current = reference_parameters(self.models)
        require(len(current) == len(self.parameters), 'reference_parameter_count_changed')
        for (name, param), (expected_name, expected), spec in zip(current, self.parameters, self.specs):
            require(
                name == expected_name and param is expected
                and (id(param), param.dtype, tuple(param.shape), tuple(param.stride()), param._version) == spec[:5],
                'reference_identity_or_version_changed')
            require(str(param.device) == (spec[5] if placement == 'original' else 'cpu'), 'reference_placement_changed')

    def prepare(self):
        require(not self.ready and self.fd is None, 'cache_prepare_once')
        require(self.parent_fd is not None, 'cache_closed')
        try:
            return self._prepare()
        except BaseException as error:
            # No parameter is rebound until offload(), so failed preparation
            # can discard its mappings. Drop traceback-held local Tensor views
            # before closing mmap and file descriptors.
            error.__traceback__ = None
            self._discard_incomplete()
            raise error from None

    def _discard_incomplete(self):
        self.views.clear()
        if self.map is not None:
            self.map.close()
            self.map = None
        for name in ('fd', 'dir_fd', 'parent_fd'):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)
        self.ready = False

    def _prepare(self):
        import torch
        self._validate(placement='original')
        page = mmap.PAGESIZE
        sizes = [p.numel() * p.element_size() for _, p in self.parameters]
        offsets, total = [], 0
        for size in sizes:
            offsets.append(total)
            total += ((size + page - 1) // page) * page
        require(total > 0, 'reference_all_empty')
        self.expected_file_bytes = total
        space = os.fstatvfs(self.parent_fd)
        require(space.f_bavail * space.f_frsize >= self.capacity_replicas * (total + max(CHUNK_BYTES, total // 20)),
                'scratch_insufficient_space')
        os.mkdir(self.cache_name, mode=0o700, dir_fd=self.parent_fd)
        self.dir_fd = os.open(self.cache_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.parent_fd)
        self.fd = os.open('weights.bin', os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.dir_fd)
        os.posix_fallocate(self.fd, 0, total)
        self.map = mmap.mmap(self.fd, total, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        # Every CPU view directly uses mapped file storage. There is no full
        # reference CPU clone, .to(cpu), pin_memory or intermediate byte array.
        for (name, param), offset, size in zip(self.parameters, offsets, sizes):
            view = (
                torch.frombuffer(self.map, dtype=param.dtype, count=param.numel(), offset=offset).view(param.shape)
                if size else torch.empty_like(param, device='cpu'))
            require(not view.is_pinned(), 'mapped_destination_pinned')
            flat, source = view.reshape(-1), param.detach().reshape(-1)
            width = max(1, self.chunk_bytes // param.element_size())
            with torch.no_grad():
                for start in range(0, param.numel(), width):
                    end = min(start + width, param.numel())
                    flat[start:end].copy_(source[start:end], non_blocking=False)
                    # Blocking D2H completes before flush/reclaim. Flush before
                    # reclaim so dirty pages cannot accumulate a full model.
                    first_page = (offset + start * param.element_size()) // page * page
                    last_page = ((offset + end * param.element_size() + page - 1) // page) * page
                    self.map.flush(first_page, last_page - first_page)
                    os.fdatasync(self.fd)
                    self.map.madvise(mmap.MADV_DONTNEED, first_page, last_page - first_page)
                    self._advise_file_reclaim(first_page, last_page - first_page)
            self.views.append(view)
            del flat, source
        self._validate(placement='original')
        self.map.flush()
        os.fdatasync(self.fd)
        self.file_identity = self._file_stat()
        initial_hash_started = time.monotonic()
        self.content_sha256, self.file_identity = self._file_digest(self.file_identity)
        self.initial_content_hash_seconds = time.monotonic() - initial_hash_started
        self.view_versions = tuple(view._version for view in self.views)
        metadata = dict(
            self.identity,
            filesystem=self.filesystem,
            bytes=total,
            logical_tensor_bytes=sum(sizes),
            parameters=len(sizes),
            parameter_inventory_sha256=self.inventory_sha256,
            parameter_inventory_scope='supplied_objects_not_checkpoint_attestation',
            complete=True,
            logical_immutable=True,
            OS_readonly_mapping=False,
            content_sha256=self.content_sha256,
            file_identity=self.file_identity,
            content_verification='complete_file_sha256_each_reuse',
            maximum_read_buffer_bytes=VERIFY_BYTES)
        meta = os.open('complete.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.dir_fd)
        with os.fdopen(meta, 'w', encoding='utf-8') as stream:
            json.dump(metadata, stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(self.dir_fd)
        os.fsync(self.parent_fd)
        self.ready = True
        return metadata

    def offload(self):
        require(self.ready, 'cache_not_complete')
        self._validate(placement='original')
        self._verify_contents('offload')
        for (_, parameter), view in zip(self.parameters, self.views):
            parameter.data = view
        self._validate(placement='cpu')
        return {
            'parameters': len(self.parameters),
            'reference_D2H_copies': 0,
            'mapped_logical_bytes': sum(v.numel() * v.element_size() for v in self.views),
            'physical_memory_savings_measured': False
        }

    def after_restore(self, *, copies_completed=False):
        require(self.ready, 'cache_not_complete')
        self._validate(placement='original')
        require(copies_completed is True, 'restore_copy_completion_required')
        # Drain only this owned file after native H2D completion. Do not bless
        # timestamp changes without full bytes, or drop unrelated page cache.
        self.map.flush()
        os.fdatasync(self.fd)
        observed = self._file_stat()
        event = dict(
            rank=self.identity['rank'],
            pid=os.getpid(),
            phase='after_restore_file_drain',
            before=self.file_identity,
            after=observed,
            content_verified_at_this_boundary=False,
            full_content_verification_required_before_next_offload=True)
        self.last_restore_observation = event
        logger.debug('Reference cache restore drain: %s', json.dumps(event, sort_keys=True))
        # H2D must complete before advice. Logical storage capacity remains;
        # successful advice does not measure physical RAM/HBM savings.
        self.map.madvise(mmap.MADV_DONTNEED)
        self._advise_file_reclaim(0, 0)
        return {
            'parameters': len(self.parameters),
            'identity_preserved': True,
            'cache_reclaim_advised_after_H2D': True,
            'owned_file_pagecache_reclaim_advised': True,
            'physical_memory_savings_measured': False
        }

    def _advise_file_reclaim(self, offset, length):
        # MADV_DONTNEED drops mappings; this separate advisory request also
        # targets clean page cache belonging only to our owned file descriptor.
        # Neither operation proves that the kernel reclaimed physical pages.
        require(hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_DONTNEED'), 'file_reclaim_advice_unavailable')
        os.posix_fadvise(self.fd, offset, length, os.POSIX_FADV_DONTNEED)


def validate_reference_cache_args(args, *, pre_init=False):
    """Validate opt-in scope without importing Torch or allocating model state."""
    directory = getattr(args, 'reference_model_cache_dir', None)
    allow_nfs = getattr(args, 'reference_model_cache_allow_nfs', False)
    release = getattr(args, 'release_model_cpu_cache', False)
    require(type(allow_nfs) is bool and type(release) is bool, 'option_type')
    require(not allow_nfs or directory is not None, 'nfs_requires_cache_directory')
    if directory is None and not release:
        return False
    require(
        getattr(args, 'rlhf_type', None) == 'grpo' and getattr(args, 'tuner_type', None) == 'full'
        and getattr(args, 'offload_model', False) is True, 'full_grpo_offload_required')
    if directory is None:
        return False
    require(type(directory) is str and bool(directory) and Path(directory).is_absolute(), 'cache_directory_invalid')
    require(
        getattr(args, 'use_vllm', False) is True and getattr(args, 'vllm_mode', None) == 'colocate',
        'colocated_vllm_required')
    # The native restore utility targets CUDA; do not silently reinterpret a
    # CPU-initialized reference's original placement.
    require(not getattr(args, 'use_cpu_initialization', False), 'cpu_initialization_unsupported')
    require(not getattr(args, 'sync_ref_model', False), 'mutable_reference_unsupported')
    beta = getattr(args, 'beta', None)
    import math
    if pre_init and beta is None:
        return True  # Native GRPO argument initialization supplies its default.
    require(type(beta) in (int, float) and math.isfinite(beta) and beta >= 0, 'beta_invalid')
    if beta == 0:
        require(
            not any(
                getattr(args, key, None)
                for key in ('ref_model', 'mcore_ref_model', 'ref_adapters', 'mcore_ref_adapter')),
            'kl_free_reference_conflict')
    return True


def reject_reference_aliases(reference_models, other_models):
    """Reject shared modules, parameters or backing storage before file writes."""
    require(not {id(m) for m in reference_models} & {id(m) for m in other_models}, 'model_alias')
    reference = reference_parameters(reference_models)
    other_parameters = [p for m in other_models for p in m.parameters()]
    require(not {id(p) for _, p in reference} & {id(p) for p in other_parameters}, 'parameter_alias')
    keys = {(str(p.device), p.untyped_storage()._cdata) for _, p in reference if p.numel()}
    others = {(str(p.device), p.untyped_storage()._cdata) for p in other_parameters if p.numel()}
    require(not keys & others, 'storage_alias')


def _validate_local_size(world_size, local_world_size):
    require(type(world_size) is int and world_size > 0, 'world_size_invalid')
    require(
        type(local_world_size) is int and 0 < local_world_size <= world_size and world_size % local_world_size == 0,
        'local_world_size_invalid')


def reference_cache_local_world_size(args, world_size):
    """Use explicit launch metadata, never a guessed GPU count or default one."""
    require(type(world_size) is int and world_size > 0, 'world_size_invalid')
    settings = [os.environ[key] for key in ('LOCAL_WORLD_SIZE', 'LOCAL_SIZE') if key in os.environ]
    require(settings or world_size == 1, 'local_world_size_missing')
    parsed = []
    for value in settings:
        require(
            type(value) is str and len(value) <= 9 and re.fullmatch('[1-9][0-9]*', value), 'local_world_size_invalid')
        parsed.append(int(value))
    require(not parsed or len(set(parsed)) == 1, 'local_world_size_conflict')
    local = parsed[0] if parsed else 1
    _validate_local_size(world_size, local)
    configured = getattr(args, 'local_world_size', None)
    require((configured is None and world_size == 1) or (type(configured) is int and configured == local),
            'local_world_size_args_mismatch')
    return local
