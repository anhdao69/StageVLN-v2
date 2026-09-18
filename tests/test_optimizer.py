import tempfile
import unittest
from types import SimpleNamespace
import torch
from transformers import TrainingArguments
from qwen_vl.train.trainer import QwenSFTTrainer

class OptimizerTests(unittest.TestCase):
    def test_custom_norm_decay_and_merger_lr(self):
        class CustomNorm(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(4))
        model=torch.nn.Module();model.model=torch.nn.Module()
        model.model.visual=torch.nn.Module()
        model.model.visual.encoder=torch.nn.Linear(4,4)
        model.model.visual.encoder.requires_grad_(False)
        model.model.visual.merger=torch.nn.Linear(4,4)
        model.language=torch.nn.Linear(4,4)
        model.custom_norm=CustomNorm()
        trainer=object.__new__(QwenSFTTrainer);trainer.model=model;trainer.optimizer=None
        trainer.args=SimpleNamespace(learning_rate=1e-6,mm_projector_lr=1e-5,weight_decay=0.01)
        trainer.get_optimizer_cls_and_kwargs=lambda args:(torch.optim.AdamW,{'lr':args.learning_rate})
        opt=trainer.create_optimizer()
        group_by_id={id(p):g for g in opt.param_groups for p in g['params']}
        self.assertNotIn(id(model.model.visual.encoder.weight),group_by_id)
        self.assertEqual(group_by_id[id(model.custom_norm.weight)]['weight_decay'],0.)
        self.assertEqual(group_by_id[id(model.language.bias)]['weight_decay'],0.)
        self.assertEqual(group_by_id[id(model.language.weight)]['weight_decay'],0.01)
        self.assertEqual(group_by_id[id(model.model.visual.merger.weight)]['lr'],1e-5)
        self.assertEqual(group_by_id[id(model.language.weight)]['lr'],1e-6)
