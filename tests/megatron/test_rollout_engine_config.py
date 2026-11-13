# Copyright (c) ModelScope Contributors. All rights reserved.
"""Run the actual engine-construction method with CPU-only dependency stand-ins."""
import ast
from copy import copy, deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch


def load_engine_method():
    source = Path(__file__).resolve().parents[2] / 'swift/megatron/trainers/rollout_mixin.py'
    tree = ast.parse(source.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MegatronRolloutMixin')
    method = next(
        node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == '_prepare_vllm_engine')
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {'copy': copy}
    # This is the checkout's actual method, not a test copy of its implementation.
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace['_prepare_vllm_engine']


class RolloutEngineConfigTest(TestCase):

    def trainer(self, engine_kwargs):
        args = SimpleNamespace(
            micro_batch_size=1,
            vllm_max_num_seqs=16,
            vllm_engine_kwargs=engine_kwargs,
            router_replay_mode=None,
            lora_rank=8,
            tuner_type='full',
            model_info=SimpleNamespace(model_dir='synthetic-model'),
            torch_dtype='bfloat16',
            model_type='synthetic',
            vllm_enable_prefix_caching=True,
            vllm_enforce_eager=True,
            vllm_limit_mm_per_prompt=None,
            sleep_level=1,
            vllm_max_model_len=65536,
            vllm_disable_cascade_attn=False,
            vllm_mm_processor_cache_gb=0)
        return SimpleNamespace(
            args=args,
            template=SimpleNamespace(padding_free=True, sequence_parallel_size=8),
            vllm_version_ge_0_10_2=True,
            vllm_tensor_parallel_size=4,
            vllm_gpu_memory_utilization=0.4,
            process_index=0)

    def invoke(self, trainer, constructed):
        inference = ModuleType('swift.infer_engine')

        def engine(*args, **kwargs):
            constructed.append(kwargs)
            return SimpleNamespace(arguments=kwargs)

        inference.GRPOVllmEngine = engine
        vllm = ModuleType('vllm')
        distributed = ModuleType('vllm.distributed')
        distributed.parallel_state = SimpleNamespace(get_tp_group=lambda: SimpleNamespace(device_group='tp4'))
        with patch.dict('sys.modules', {
                'vllm': vllm,
                'vllm.distributed': distributed,
                'swift.infer_engine': inference
        }):
            return load_engine_method()(trainer)

    def test_enabled_expert_parallel_reaches_constructor_without_mutating_arguments(self):
        original = {'enable_expert_parallel': True, 'load_format': 'auto', 'max_num_batched_tokens': 8192}
        trainer = self.trainer(deepcopy(original))
        constructed = []
        for _ in range(2):
            result = self.invoke(trainer, constructed)
            self.assertIs(result.arguments['enable_expert_parallel'], True)
            self.assertEqual(result.arguments['engine_kwargs'], {'max_num_batched_tokens': 8192})
            self.assertEqual(result.arguments['max_num_seqs'], 16)
            self.assertEqual(trainer.args.vllm_engine_kwargs, original)
            self.assertTrue(trainer.template.padding_free)
            self.assertEqual(trainer.template.sequence_parallel_size, 8)
            self.assertEqual(trainer.vllm_tp_group, 'tp4')
        self.assertEqual(len(constructed), 2)

    def test_disabled_and_default_settings_remain_disabled(self):
        for values in (None, {}, {'enable_expert_parallel': False}):
            with self.subTest(values=values):
                result = self.invoke(self.trainer(values), [])
                self.assertIs(result.arguments['enable_expert_parallel'], False)

    def test_invalid_boolean_is_rejected_before_engine_construction(self):
        for value in (1, 0, 'true', 'false', None, []):
            trainer = self.trainer({'enable_expert_parallel': value, 'load_format': 'auto'})
            before = deepcopy(trainer.args.vllm_engine_kwargs)
            constructed = []
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'must be a boolean'):
                self.invoke(trainer, constructed)
            self.assertEqual(constructed, [])
            self.assertEqual(trainer.args.vllm_engine_kwargs, before)


if __name__ == '__main__':
    main()
