# ruff: noqa: E402
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import decord  # noqa: F401  # import after torch and before torchvision
import transformers
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwen_vl.train.sampler  # noqa: F401  # installs the custom VLN sampler
from qwen_vl.train.trainer import SpatialForcingTrainer

from qwen_vl.data.data_qwen import make_supervised_data_module

from qwen_vl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import (
    AutoConfig,
    AutoProcessor,
    set_seed,
)
from transformers.utils.hub import cached_file

local_rank = None
QWEN3_5_MODEL_TYPES = {"qwen3_5", "qwen3_5_vl"}


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def resolve_model_modules(model):
    if hasattr(model, "visual") and hasattr(model, "model"):
        return (
            model.visual,
            getattr(model.visual, "merger", None),
            model.model,
            model.lm_head,
        )

    inner_model = getattr(model, "model", None)
    if (
        inner_model is not None
        and hasattr(inner_model, "visual")
        and hasattr(inner_model, "language_model")
    ):
        return (
            inner_model.visual,
            getattr(inner_model.visual, "merger", None),
            inner_model.language_model,
            model.lm_head,
        )

    raise ValueError(f"Unsupported model structure for training: {type(model)}")


def set_model(model_args, model):
    visual_module, merger_module, language_module, lm_head = resolve_model_modules(
        model
    )

    if model_args.tune_mm_vision:
        for n, p in visual_module.named_parameters():
            p.requires_grad = True
    else:
        for n, p in visual_module.named_parameters():
            p.requires_grad = False

    if merger_module is not None:
        if model_args.tune_mm_mlp:
            for n, p in merger_module.named_parameters():
                p.requires_grad = True
        else:
            for n, p in merger_module.named_parameters():
                p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in language_module.named_parameters():
            p.requires_grad = True
        for p in lm_head.parameters():
            p.requires_grad = True
    else:
        for n, p in language_module.named_parameters():
            p.requires_grad = False
        for p in lm_head.parameters():
            p.requires_grad = False

    if getattr(model, "spatial_teacher", None) is None:
        raise ValueError("Spatial Forcing teacher was not initialized")
    model.spatial_teacher.requires_grad_(False)
    model.spatial_teacher.eval()
    if getattr(model, "spatial_projector", None) is None:
        raise ValueError("Spatial Forcing projector was not initialized")
    model.spatial_projector.requires_grad_(True)
    if model_args.depth_supervision_enabled:
        if getattr(model, "student_depth_head", None) is None:
            raise ValueError("Depth supervision head was not initialized")
        model.student_depth_head.requires_grad_(True)


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    # enable_full_determinism(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model_type = getattr(config, "model_type", None)

    if (
        model_type not in QWEN3_5_MODEL_TYPES
        and "qwen3.5" not in model_args.model_name_or_path.lower()
    ):
        raise ValueError("SpatialForcing-VLN supports Qwen3.5 only")
    if not model_args.spatial_forcing_enabled:
        raise ValueError(
            "This training entry point requires spatial_forcing_enabled=True"
        )
    if model_args.use_geometry_encoder:
        raise ValueError("Set use_geometry_encoder=False: VGGT is a loss-only teacher")
    if model_args.use_geometry_fusion:
        raise ValueError("Set use_geometry_fusion=False for Spatial Forcing")
    if model_args.sf_multiframe_teacher:
        raise ValueError("v4 depth supervision requires sf_multiframe_teacher=False")
    if model_args.depth_supervision_enabled:
        if model_args.depth_student_layers != [7, 16, 24, 32]:
            raise ValueError(
                "v4 requires depth_student_layers=[7, 16, 24, 32], got "
                f"{model_args.depth_student_layers}"
            )
        if model_args.depth_loss_type != "geo_depth":
            raise ValueError("v4 supports only depth_loss_type=geo_depth")
        if model_args.depth_use_teacher_confidence:
            raise ValueError("v4 intentionally excludes teacher-confidence weighting")
    if data_args.data_flatten:
        raise ValueError("Spatial Forcing requires data_flatten=False")

    from qwen_vl.model.spatial_forcing import (
        Qwen3_5ForConditionalGenerationWithSpatialForcing,
    )

    for key in [
        "use_geometry_encoder",
        "use_geometry_fusion",
        "spatial_forcing_enabled",
        "sf_loss_weight",
        "sf_student_layer",
        "sf_teacher_layer",
        "sf_use_vggt_pe",
        "sf_projector_hidden_dim",
        "sf_verify_invariants",
        "sf_multiframe_teacher",
        "depth_supervision_enabled",
        "depth_loss_weight",
        "depth_student_layers",
        "depth_loss_type",
        "depth_gradient_scales",
        "depth_outlier_keep_ratio",
        "depth_use_teacher_confidence",
    ]:
        setattr(config, key, getattr(model_args, key))
    config.sf_teacher_dim = 2048
    model = Qwen3_5ForConditionalGenerationWithSpatialForcing.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        spatial_teacher_model_path=model_args.geometry_encoder_path,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
    )
    data_args.image_processor = processor.image_processor
    data_args.processor = processor
    data_args.model_type = "qwen3.5"
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    import torch.distributed as dist

    def is_rank_zero():
        return (
            (not dist.is_available())
            or (not dist.is_initialized())
            or dist.get_rank() == 0
        )

    if is_rank_zero():
        visual_module, _, language_module, _ = resolve_model_modules(model)
        visual_module.print_trainable_parameters()
        language_module.print_trainable_parameters()

    if is_rank_zero():
        print(model.config)
    setattr(data_args, "use_geometry_encoder", False)
    setattr(data_args, "spatial_forcing_enabled", True)
    setattr(
        data_args,
        "depth_supervision_enabled",
        model_args.depth_supervision_enabled,
    )
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = SpatialForcingTrainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    if getattr(data_args, "processor", None) is not None:
        data_args.processor.save_pretrained(training_args.output_dir)
    else:
        data_args.image_processor.save_pretrained(training_args.output_dir)

    template_filename = "chat_template.json"
    template_path = os.path.join(training_args.output_dir, template_filename)

    source_path = None
    if os.path.isdir(model_args.model_name_or_path):
        candidate_path = os.path.join(model_args.model_name_or_path, template_filename)
        if os.path.isfile(candidate_path):
            source_path = candidate_path
    else:
        try:
            source_path = cached_file(
                model_args.model_name_or_path,
                template_filename,
                cache_dir=training_args.cache_dir,
            )
        except (OSError, EnvironmentError) as err:
            if getattr(data_args, "processor", None) is None:
                logging.warning(
                    "Unable to locate %s for model %s: %s",
                    template_filename,
                    model_args.model_name_or_path,
                    err,
                )

    if source_path:
        if os.path.abspath(source_path) != os.path.abspath(template_path):
            shutil.copy2(source_path, template_path)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
