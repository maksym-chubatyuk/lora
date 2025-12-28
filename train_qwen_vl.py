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
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


MODEL_ID = "Qwen/Qwen3-8B"

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
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-5,
    "num_train_epochs": 3,
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "logging_steps": 10,
    "save_steps": 100,
    "save_total_limit": 3,
    "bf16": False,
    "fp16": True,
    "gradient_checkpointing": True,
    "optim": "adamw_torch",
    "lr_scheduler_type": "linear",
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


def format_chat(example: dict, tokenizer) -> dict:
    """Format a single example for training."""
    messages = example["messages"]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}


def tokenize_function(examples: dict, tokenizer, max_length: int) -> dict:
    """Tokenize examples for training."""
    tokenized = tokenizer(
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
        default=4,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
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

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model in fp16
    print("Loading model in fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

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
        lambda x: format_chat(x, tokenizer),
        desc="Formatting"
    )

    # Tokenize
    print("Tokenizing...")
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
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
        fp16=DEFAULT_TRAINING_CONFIG["fp16"],
        gradient_checkpointing=DEFAULT_TRAINING_CONFIG["gradient_checkpointing"],
        optim=DEFAULT_TRAINING_CONFIG["optim"],
        lr_scheduler_type=DEFAULT_TRAINING_CONFIG["lr_scheduler_type"],
        report_to=DEFAULT_TRAINING_CONFIG["report_to"],
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
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
    tokenizer.save_pretrained(args.output)

    print(f"\nTraining complete! LoRA adapters saved to: {args.output}")
    print("\nNext steps:")
    print("  1. Run export_model.py to merge and quantize for inference")
    print("  2. Or load directly with PEFT for inference")


if __name__ == "__main__":
    main()
