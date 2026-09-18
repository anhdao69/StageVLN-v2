"""Plain supervised fine-tuning for Qwen/Qwen3.5-4B on JanusVLN R2R."""

import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

import qwen_vl.train.sampler  # noqa: F401 - installs length-grouped sampling
from qwen_vl.data.data_qwen import make_supervised_data_module
from qwen_vl.train.argument import DataArguments, ModelArguments, TrainingArguments
from qwen_vl.train.trainer import QwenSFTTrainer


def _model_parts(model):
    return (
        model.model.visual,
        model.model.visual.merger,
        model.model.language_model,
        model.lm_head,
    )


def _set_trainable_modules(model, model_args):
    vision, merger, language_model, lm_head = _model_parts(model)
    vision.requires_grad_(model_args.tune_mm_vision)
    merger.requires_grad_(model_args.tune_mm_mlp)
    language_model.requires_grad_(model_args.tune_mm_llm)
    lm_head.requires_grad_(model_args.tune_mm_llm)


def _print_trainable_summary(model):
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    vision, merger, language_model, lm_head = _model_parts(model)
    vision_trainable = sum(
        p.numel() for p in vision.parameters() if p.requires_grad
    ) - sum(p.numel() for p in merger.parameters() if p.requires_grad)
    vision_total = sum(p.numel() for p in vision.parameters()) - sum(
        p.numel() for p in merger.parameters()
    )
    print(
        f">>>>> vision backbone: {vision_trainable:,}/{vision_total:,} "
        "trainable parameters"
    )
    for name, module in (
        ("vision merger", merger),
        ("language model", language_model),
        ("lm head", lm_head),
    ):
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in module.parameters())
        print(f">>>>> {name}: {trainable:,}/{total:,} trainable parameters")


def _save_model(trainer, output_dir):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    if trainer.args.should_save:
        state_dict = {
            name: value.cpu() for name, value in trainer.model.state_dict().items()
        }
        trainer._save(output_dir, state_dict=state_dict)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    transformers.set_seed(training_args.seed)
    os.makedirs(training_args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    if config.model_type not in {"qwen3_5", "qwen3_5_vl"}:
        raise ValueError(
            f"Expected a Qwen3.5 model, got model_type={config.model_type!r}"
        )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
        dtype=torch.bfloat16 if training_args.bf16 else None,
    )
    model.config.use_cache = False
    model.model.language_model.config.use_cache = False
    _set_trainable_modules(model, model_args)
    _print_trainable_summary(model)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
    )
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = training_args.model_max_length
    tokenizer.padding_side = "right"
    data_args.processor = processor

    data_module = make_supervised_data_module(tokenizer, data_args)
    trainer = QwenSFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    checkpoints = list(Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        logging.info("Resuming from the latest checkpoint in %s", training_args.output_dir)
    train_result = trainer.train(resume_from_checkpoint=True if checkpoints else None)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if training_args.should_save:
        processor.save_pretrained(training_args.output_dir)
    if training_args.save_final_model:
        model.config.use_cache = True
        model.model.language_model.config.use_cache = True
        _save_model(trainer, training_args.output_dir)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
