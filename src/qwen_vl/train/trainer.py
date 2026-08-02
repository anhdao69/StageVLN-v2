"""Trainer extensions for Spatial Forcing metrics and optimizer groups."""

import torch
from transformers import Trainer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextModel,
    Qwen3_5VisionModel,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import get_parameter_names


def print_trainable_parameters_visual(self) -> None:
    trainable_blocks = []
    frozen_blocks = []
    for block_index, block in enumerate(self.blocks):
        target = (
            trainable_blocks
            if all(parameter.requires_grad for parameter in block.parameters())
            else frozen_blocks
        )
        target.append(block_index)

    merger_trainable = any(
        parameter.requires_grad for parameter in self.merger.parameters()
    )
    print("Vision Module - Attention Blocks:")
    print(f"Trainable Block Indices: {trainable_blocks or 'None'}")
    print(f"Non-Trainable Block Indices: {frozen_blocks or 'None'}")
    print(f"Merger Module Trainable: {merger_trainable}")


def print_trainable_parameters(self) -> None:
    embeddings_trainable = any(
        parameter.requires_grad for parameter in self.embed_tokens.parameters()
    )
    trainable_layers = []
    frozen_layers = []
    for layer_index, layer in enumerate(self.layers):
        target = (
            trainable_layers
            if any(parameter.requires_grad for parameter in layer.parameters())
            else frozen_layers
        )
        target.append(layer_index)

    print(f"LLM Module - Embed Tokens Trainable: {embeddings_trainable}")
    print(f"LLM Module - Trainable Layer Indices: {trainable_layers or 'None'}")
    print(f"LLM Module - Non-Trainable Layer Indices: {frozen_layers or 'None'}")


class SpatialForcingTrainer(Trainer):
    """Report the task and alignment objectives separately."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spatial_metric_sums = {}
        self._spatial_metric_count = 0
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        unwrapped_model = self.accelerator.unwrap_model(model)
        metrics = getattr(unwrapped_model, "last_spatial_forcing_metrics", {})
        if metrics:
            for name, value in metrics.items():
                scalar = float(value.detach().float().mean().item())
                self._spatial_metric_sums[name] = (
                    self._spatial_metric_sums.get(name, 0.0) + scalar
                )
            self._spatial_metric_count += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if "loss" in logs and self._spatial_metric_count:
            metric_names = sorted(self._spatial_metric_sums)
            packed = torch.tensor(
                [self._spatial_metric_sums[name] for name in metric_names]
                + [float(self._spatial_metric_count)],
                dtype=torch.float64,
                device=self.args.device,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            total_count = max(float(packed[-1].item()), 1.0)
            for metric_index, metric_name in enumerate(metric_names):
                logs[metric_name] = float(packed[metric_index].item() / total_count)

            if "spatial_forcing_loss" in logs:
                logs["sf_loss"] = logs["spatial_forcing_loss"]
            if "mean_cosine_similarity" in logs:
                logs["cosine_similarity"] = logs["mean_cosine_similarity"]

            unwrapped_model = self.accelerator.unwrap_model(self.model)
            logs["projector_grad_verified"] = float(
                getattr(unwrapped_model, "_sf_projector_grad_verified", False)
            )
            logs["student_grad_verified"] = float(
                getattr(unwrapped_model, "_sf_student_grad_verified", False)
            )
            teacher = getattr(unwrapped_model, "spatial_teacher", None)
            logs["vggt_has_gradient"] = float(
                teacher is not None
                and any(
                    parameter.grad is not None for parameter in teacher.parameters()
                )
            )
            self._spatial_metric_sums = {}
            self._spatial_metric_count = 0
        super().log(logs, start_time=start_time)


def create_optimizer(self):
    """Use the configured projector LR for Qwen's merger and SF projector."""
    if self.optimizer is not None:
        return self.optimizer

    decay_names = set(get_parameter_names(self.model, ALL_LAYERNORM_LAYERS))
    decay_names = {name for name in decay_names if "bias" not in name}
    projector_names = {
        name
        for name, _ in self.model.named_parameters()
        if "merger" in name or "spatial_projector" in name
    }
    projector_lr = self.args.mm_projector_lr

    def parameters(*, decay: bool, projector: bool):
        return [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and (name in decay_names) is decay
            and (name in projector_names) is projector
        ]

    grouped_parameters = []
    for use_decay in (True, False):
        base_parameters = parameters(decay=use_decay, projector=False)
        if base_parameters:
            grouped_parameters.append(
                {
                    "params": base_parameters,
                    "weight_decay": self.args.weight_decay if use_decay else 0.0,
                }
            )

    for use_decay in (True, False):
        projected_parameters = parameters(decay=use_decay, projector=True)
        if projected_parameters:
            group = {
                "params": projected_parameters,
                "weight_decay": self.args.weight_decay if use_decay else 0.0,
            }
            if projector_lr is not None and projector_lr != 0:
                group["lr"] = projector_lr
            grouped_parameters.append(group)

    optimizer_class, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
    self.optimizer = optimizer_class(grouped_parameters, **optimizer_kwargs)
    return self.optimizer


Trainer.create_optimizer = create_optimizer
Qwen3_5VisionModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen3_5TextModel.print_trainable_parameters = print_trainable_parameters
