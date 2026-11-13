# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import Literal, Optional

from .sft_args import MegatronSftArguments


@dataclass
class MegatronRLHFArguments(MegatronSftArguments):
    rlhf_type: Literal['dpo', 'kto', 'grpo', 'gkd', 'rm'] = 'dpo'
    loss_scale: str = 'last_round'
    truncation_strategy: Optional[Literal['delete', 'left', 'right', 'split', None]] = None

    calculate_per_token_loss: bool = False
    reference_model_cache_dir: Optional[str] = None
    reference_model_cache_allow_nfs: bool = False
    release_model_cpu_cache: bool = False

    def __post_init__(self):
        memory_options = (
            self.reference_model_cache_dir is not None or self.reference_model_cache_allow_nfs
            or self.release_model_cpu_cache)
        if memory_options:
            from swift.megatron.trainers.reference_cache import validate_reference_cache_args
            validate_reference_cache_args(self, pre_init=True)
        if self.rlhf_type == 'rm':
            self.task_type = 'seq_cls'
            self.num_labels = 1
        self._init_truncation_strategy()
        super().__post_init__()
        if memory_options:
            validate_reference_cache_args(self)

    def _init_truncation_strategy(self):
        if self.truncation_strategy is not None:
            return
        if self.rlhf_type == 'grpo':
            self.truncation_strategy = 'left'
        else:
            self.truncation_strategy = 'delete'
