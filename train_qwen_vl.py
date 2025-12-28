#!/usr/bin/env python3
"""
QLoRA fine-tuning script for Qwen3-VL-8B-Thinking.

This script fine-tunes only the text generation layers while preserving
vision capabilities. Designed for A100 40GB.

Usage:
    python train_qwen_vl.py --data asuka_training_data_qwen.jsonl
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Optional

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"

# Default QLoRA configuration
DEFAULT_LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

# Default training configuration
DEFAULT_TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "logging_steps": 10,
    "save_steps": 100,
    "save_total_limit": 3,
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit",
    "lr_scheduler_type": "cosine",
    "report_to": "none",
}


def load_dataset(data_path: str) -> Dataset:
    """Load and prepare the dataset."""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return Dataset.from_list(data)


def format_chat(example: dict, processor) -> dict:
    """Format a single example for training."""
    messages = example["messages"]

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}


def tokenize_function(examples: dict, processor, max_length: int) -> dict:
    """Tokenize examples for training."""
    tokenized = processor(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    # For causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen3-VL")
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to training data (Qwen format JSONL)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./qwen3-vl-asuka-lora",
        help="Output directory for LoRA adapters"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=128,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )

    args = parser.parse_args()

    print(f"Loading model: {MODEL_ID}")
    print(f"Training data: {args.data}")
    print(f"Output: {args.output}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load processor (tokenizer + image processor)
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Load model in 4-bit
    print("Loading model in 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # Use flash attention if available
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=DEFAULT_LORA_CONFIG["lora_dropout"],
        target_modules=DEFAULT_LORA_CONFIG["target_modules"],
        bias=DEFAULT_LORA_CONFIG["bias"],
        task_type=DEFAULT_LORA_CONFIG["task_type"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    print("\nLoading dataset...")
    dataset = load_dataset(args.data)
    print(f"  Total examples: {len(dataset)}")

    # Format with chat template
    print("Formatting with chat template...")
    dataset = dataset.map(
        lambda x: format_chat(x, processor),
        desc="Formatting"
    )

    # Tokenize
    print("Tokenizing...")
    dataset = dataset.map(
        lambda x: tokenize_function(x, processor.tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=DEFAULT_TRAINING_CONFIG["warmup_ratio"],
        logging_steps=DEFAULT_TRAINING_CONFIG["logging_steps"],
        save_steps=DEFAULT_TRAINING_CONFIG["save_steps"],
        save_total_limit=DEFAULT_TRAINING_CONFIG["save_total_limit"],
        bf16=DEFAULT_TRAINING_CONFIG["bf16"],
        gradient_checkpointing=DEFAULT_TRAINING_CONFIG["gradient_checkpointing"],
        optim=DEFAULT_TRAINING_CONFIG["optim"],
        lr_scheduler_type=DEFAULT_TRAINING_CONFIG["lr_scheduler_type"],
        report_to=DEFAULT_TRAINING_CONFIG["report_to"],
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
    )

    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Save final model
    print("\nSaving LoRA adapters...")
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)

    print(f"\nTraining complete! LoRA adapters saved to: {args.output}")
    print("\nNext steps:")
    print("  1. Run export_model.py to merge and quantize for inference")
    print("  2. Or load directly with PEFT for inference")


if __name__ == "__main__":
    main()
