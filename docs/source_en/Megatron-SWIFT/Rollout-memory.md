# Megatron GRPO rollout memory

Colocated GRPO alternates training, weight export, generation and reference-model
inference. GPU utilization alone does not distinguish those phases. Model CPU
copies, optimizer state, pinned staging buffers and filesystem page cache also
contribute to host memory use.

The options below are disabled by default. They do not change rewards, the KL
coefficient, sequence lengths, parallelism or checkpoint contents/cadence.

## Frozen reference model on disk

`--reference_model_cache_dir /absolute/private/scratch` enables a process-local,
file-backed cache for the frozen reference model in full-parameter GRPO with
`--use_vllm true --vllm_mode colocate --offload_model true`.
The cache requires CUDA-initialized reference parameters; `use_cpu_initialization`
must remain `false` (the default).

Create the scratch directory before starting the job. It must be owned by the
training user, must not be writable by other users, and must not contain symlinks
in its path. Use a real disk filesystem (ext4, XFS or Btrfs), not tmpfs. NFS is
accepted only with `--reference_model_cache_allow_nfs true`; shared storage still
needs enough capacity for all ranks and may add substantial I/O latency.

The free-space check budgets one cache plus reserve per local training process
on ext4, XFS or Btrfs, and per global rank on explicitly enabled NFS. Distributed
cache creation requires an explicit `LOCAL_WORLD_SIZE` or `LOCAL_SIZE` launch
setting that matches the trainer's local process count. If both are set, they
must agree; missing or malformed settings are rejected rather than inferred from
the GPU count. For example, a 56-rank job with four processes per node budgets
four caches on each local disk and 56 on shared NFS. This conservative check uses
the currently available disk space and does not reserve capacity across ranks.

The cache is initialized on the first reference offload, after checkpoint loading.
It copies frozen weights into private mapped files in bounded chunks. Later
offloads reuse those files after content/inventory checks; restoration waits for
device copies to finish before advising reclaim of the owned file pages. This
avoids a second full anonymous CPU reference copy. Reclaim advice is not proof
that the OS reclaimed a particular amount of RAM.

Only plain, contiguous, frozen parameters with supported storage layouts are
accepted. Policy/reference aliases and unsupported layouts fail explicitly.
The policy model, teacher model and optimizer retain their existing paths. This
cache is disposable local state, not a checkpoint or a cold-restore mechanism;
another process never adopts it. The files contain model weights and must receive
the same access protection as the model. Clean up retained scratch files only
after their owning processes have terminated.

With the cache option enabled, KL-free GRPO (`--beta 0`) skips both reference
allocation and cache creation. Explicit reference checkpoint or adapter settings
must not be combined with this mode. Without the option, the existing model
preparation path is unchanged.

## Release restored policy CPU copies

`--release_model_cpu_cache true` releases redundant CPU parameter buffers after
the native policy-model restore has completed. It requires full-parameter GRPO
and model offloading. It verifies resident parameter/gradient buffers and
synchronizes device copies before dropping their CPU sources. It leaves model
values, parameter aliases, frozen parameters and optimizer state intact.

The next offload allocates CPU buffers again. This trades buffer reuse for lower
retained host memory and is not a throughput guarantee. Inactive pinned allocator
cache is released when the installed PyTorch exposes the corresponding API.

## Rollout expert parallelism

For a supported MoE model, pass a JSON boolean through the existing engine options:

```shell
--vllm_engine_kwargs '{"enable_expert_parallel": true}'
```

The value is forwarded explicitly to the colocated vLLM engine without consuming
or mutating the caller's options dictionary. A string such as `"true"` is rejected.
Rollout expert parallelism is separate from Megatron training parallelism.

## Validation scope

CPU regressions exercise cache contents and corruption rejection, ownership and
storage guards, restore/release ordering, configuration boundaries and engine
argument forwarding. These checks do not prove full-model GPU memory capacity,
throughput, distributed checkpoint recovery or training quality. Validate the
integrated code on the intended GPU topology before enabling it in a new run.

Existing running jobs continue to use their original immutable source bundles;
pushing this branch does not update them.
