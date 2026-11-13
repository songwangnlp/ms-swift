# Copyright (c) ModelScope Contributors. All rights reserved.
"""Execute the checkout's offload context without initializing GPU dependencies."""
import ast
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


def load_context(events):
    source = Path(__file__).resolve().parents[2] / 'swift/megatron/trainers/rollout_mixin.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MegatronRolloutMixin')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'offload_context')
    namespace = {'contextmanager': contextmanager}
    for name, label in (
        ('offload_megatron_model_to_cpu', 'offload'),
        ('load_megatron_model_to_gpu', 'restore'),
        ('offload_megatron_optimizer', 'optimizer_offload'),
        ('load_megatron_optimizer', 'optimizer_restore'),
        ('release_restored_model_host_cache', 'release'),
    ):
        namespace[name] = lambda value, label=label: events.append((label, value))
    # Only trusted source from this repository is compiled; no external input.
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['offload_context']


class TestRolloutMemoryContext(unittest.TestCase):

    def make_trainer(self, *, release=False, hooks=False, offload=True):
        events = []
        trainer = SimpleNamespace(
            args=SimpleNamespace(offload_model=offload, offload_optimizer=True, release_model_cpu_cache=release),
            wrapped_models='policy',
            ref_models='reference',
            optimizer='optimizer')
        if hooks:
            trainer._offload_reference_model = lambda: events.append(('reference_hook', 'offload'))
            trainer._restore_reference_model = lambda: events.append(('reference_hook', 'restore'))
        return trainer, events, load_context(events)

    def test_default_legacy_order_is_unchanged(self):
        trainer, events, context = self.make_trainer()
        with context(trainer):
            events.append(('generation', None))
        self.assertEqual(events, [('offload', 'policy'), ('offload', 'reference'), ('optimizer_offload', 'optimizer'),
                                  ('generation', None), ('restore', 'policy'), ('restore', 'reference'),
                                  ('optimizer_restore', 'optimizer')])

    def test_instance_reference_hooks_and_release_order(self):
        trainer, events, context = self.make_trainer(release=True, hooks=True)
        with context(trainer):
            events.append(('generation', None))
        self.assertEqual(events, [('offload', 'policy'), ('reference_hook', 'offload'),
                                  ('optimizer_offload', 'optimizer'), ('generation', None), ('restore', 'policy'),
                                  ('release', 'policy'), ('reference_hook', 'restore'),
                                  ('optimizer_restore', 'optimizer')])

    def test_restore_runs_when_generation_raises(self):
        trainer, events, context = self.make_trainer(release=True, hooks=True)
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            with context(trainer):
                raise RuntimeError('generation failed')
        self.assertEqual(events[-4:], [('restore', 'policy'), ('release', 'policy'), ('reference_hook', 'restore'),
                                       ('optimizer_restore', 'optimizer')])

    def test_empty_reference_reaches_opt_in_validation(self):
        trainer, events, context = self.make_trainer(hooks=True)
        trainer.ref_models = []

        def reject_empty_reference():
            raise ValueError('missing reference')

        trainer._offload_reference_model = reject_empty_reference
        with self.assertRaisesRegex(ValueError, 'missing reference'):
            with context(trainer):
                self.fail('invalid reference must not start generation')
        self.assertEqual(events, [('offload', 'policy')])

    def test_model_offload_disabled_does_not_touch_model_hooks(self):
        trainer, events, context = self.make_trainer(release=True, hooks=True, offload=False)
        with context(trainer):
            pass
        self.assertEqual(events, [('optimizer_offload', 'optimizer'), ('optimizer_restore', 'optimizer')])


if __name__ == '__main__':
    unittest.main()
