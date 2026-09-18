"""Trainer support needed by plain Qwen3.5 supervised fine-tuning."""

import torch
from transformers import Trainer
from qwen_vl.train.sampler import _get_train_sampler


class QwenSFTTrainer(Trainer):
    """Give the trainable vision merger its configured learning rate."""

    _get_train_sampler = _get_train_sampler

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We normalize over action states in the entire accumulation window.
        # Trainer must not divide again; DeepSpeed scale_wrt_gas is also False.
        self.model_accepts_loss_kwargs = True

    def _get_num_items_in_batch(self, batch_samples, device):
        count = torch.tensor(sum(batch["labels"].shape[0] for batch in batch_samples), device=device)
        return self.accelerator.gather(count).sum()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        # The collator aligns selected prediction logits with labels shifted
        # once (dummy leading label/trailing logit). Dense labels use the same
        # one-shift rule. Average tokens within each action state.
        targets = labels[:, 1:]
        valid = targets.ne(-100)
        counts = valid.sum(-1)
        if (counts == 0).any():
            raise ValueError("Every action state must have supervised targets")
        token_losses = torch.nn.functional.cross_entropy(
            logits[:, :-1][valid].float(), targets[valid], reduction="none"
        )
        rows = valid.nonzero()[:, 0]
        sums = token_losses.new_zeros(labels.shape[0]).scatter_add(0, rows, token_losses)
        per_state = sums / counts
        loss = per_state.mean() if num_items_in_batch is None else (
            per_state.sum() * self.accelerator.num_processes / num_items_in_batch
        )
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        # Exclude custom RMSNorm weights as well as standard LayerNorm/bias.
        decay_names = {
            name for name, parameter in self.model.named_parameters()
            if parameter.ndim > 1 and not name.endswith("bias")
            and "norm" not in name.lower()
        }
        merger_names = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and ".visual.merger." in name
        }

        grouped_parameters = []
        for is_merger, learning_rate in (
            (False, self.args.learning_rate),
            (True, self.args.mm_projector_lr),
        ):
            for use_decay in (True, False):
                parameters = [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad
                    and (name in merger_names) is is_merger
                    and (name in decay_names) is use_decay
                ]
                if parameters:
                    grouped_parameters.append(
                        {
                            "params": parameters,
                            "weight_decay": (
                                self.args.weight_decay if use_decay else 0.0
                            ),
                            "lr": learning_rate,
                        }
                    )

        assigned = [
            id(parameter)
            for group in grouped_parameters
            for parameter in group["params"]
        ]
        expected = {
            id(parameter)
            for parameter in self.model.parameters()
            if parameter.requires_grad
        }
        if len(assigned) != len(set(assigned)) or set(assigned) != expected:
            raise RuntimeError("Optimizer groups do not exactly cover trainable parameters")

        optimizer_class, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_class(grouped_parameters, **optimizer_kwargs)
        return self.optimizer
