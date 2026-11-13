# Copyright (c) ModelScope Contributors. All rights reserved.
"""Optional release of redundant native DDP parameter offload copies."""
import gc


def release_restored_model_host_cache(models):
    """Detach CPU parameter caches after a complete native GPU restoration.

    The caller must finish restoring model and gradient buffers before calling
    this helper. Pending H2D copies are synchronized before their CPU sources
    are detached. Parameter/gradient storage, frozen parameters and optimizer
    state are never changed. Plain reference models have no DDP cache and must
    not be passed here.

    Returns the logical tensor bytes detached, not measured physical memory
    released. Other owners may still hold those CPU tensors. Older Torch
    versions without the public inactive host-cache API remain supported.
    """
    import torch
    from megatron.core.distributed import DistributedDataParallel

    parameters = []
    devices = []
    seen = set()
    released_bytes = 0
    for model in models:
        if not isinstance(model, DistributedDataParallel):
            raise ValueError('Host-cache release requires native DDP buffers')
        for buffer in [*model.buffers, *model.expert_parallel_buffers]:
            params, grads = buffer.param_data, buffer.grad_data
            cache = getattr(params, 'cpu_data', None)
            if cache is None or id(params) in seen:
                continue
            if (not params.is_cuda or not grads.is_cuda or params.device != grads.device or cache.device.type != 'cpu'
                    or cache.shape != params.shape or cache.dtype != params.dtype or params.numel() == 0
                    or grads.numel() == 0
                    or params.untyped_storage().nbytes() != params.numel() * params.element_size()
                    or grads.untyped_storage().nbytes() != grads.numel() * grads.element_size()):
                raise ValueError('Model and gradient buffers must be fully restored before cache release')
            seen.add(id(params))
            parameters.append(params)
            if params.device not in devices:
                devices.append(params.device)
            released_bytes += cache.numel() * cache.element_size()
    cache = None
    if not parameters:
        return 0
    # Native restoration can enqueue nonblocking copies on multiple streams.
    for device in devices:
        torch.cuda.synchronize(device)
    for params in parameters:
        del params.cpu_data
    gc.collect()
    empty_host_cache = getattr(getattr(torch, 'accelerator', None), 'empty_host_cache', None)
    if callable(empty_host_cache):
        empty_host_cache()
    return released_bytes
