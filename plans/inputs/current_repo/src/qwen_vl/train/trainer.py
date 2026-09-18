"""Trainer support needed by plain Qwen3.5 supervised fine-tuning."""

from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import get_parameter_names


class QwenSFTTrainer(Trainer):
    """Give the trainable vision merger its configured learning rate."""

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        decay_names = set(get_parameter_names(self.model, ALL_LAYERNORM_LAYERS))
        decay_names = {name for name in decay_names if not name.endswith("bias")}
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
