#!/usr/bin/env python3
"""
Fine-tune Qwen 8B using PyTorch + PEFT LoRA.
Trains on ShareGPT format data and saves adapters.
Optimized for A100 40GB with fast iteration times.
"""

import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
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


class DataCollatorForCausalLM:
    """Custom data collator that properly pads input_ids, attention_mask, and labels."""

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # Find max length in batch
        max_length = max(len(f["input_ids"]) for f in features)

        # Round up to multiple for tensor core efficiency
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1)
                          // self.pad_to_multiple_of * self.pad_to_multiple_of)

        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            seq_len = len(input_ids)
            padding_len = max_length - seq_len

            # Pad input_ids with pad_token_id
            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
            # Attention mask: 1 for real tokens, 0 for padding
            attention_mask = [1] * seq_len + [0] * padding_len
            # Labels: -100 for padding (ignored in loss)
            padded_labels = labels + [-100] * padding_len

            batch["input_ids"].append(padded_input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(padded_labels)

        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch


# =============================================================================
# Configuration
# =============================================================================

MODEL = "Qwen/Qwen3-8B"
DATA_FILE = "asuka_training_data.jsonl"
OUTPUT_DIR = "output/adapters"

# Training hyperparameters
MAX_STEPS = 500
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 16
LEARNING_RATE = 2e-5
WARMUP_STEPS = 30
LOGGING_STEPS = 10
SAVE_STEPS = 100
MAX_SEQ_LENGTH = 2048

# LoRA Configuration
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def check_data():
    """Verify training data exists."""
    data_file = Path(DATA_FILE)
    if not data_file.exists():
        print(f"Error: {DATA_FILE} not found!")
        return False

    # Count examples
    with open(data_file) as f:
        count = sum(1 for _ in f)
    print(f"Training examples: {count}")
    return True


def load_training_data(tokenizer):
    """Load and preprocess ShareGPT format training data."""
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    def format_and_tokenize(example):
        """Convert ShareGPT format to chat template and tokenize."""
        messages = []
        for turn in example["conversations"]:
            role_map = {"system": "system", "human": "user", "gpt": "assistant"}
            role = role_map.get(turn["from"], turn["from"])
            messages.append({
                "role": role,
                "content": turn["value"]
            })

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        result = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    # Process dataset
    dataset = dataset.map(
        format_and_tokenize,
        remove_columns=["conversations"],
        desc="Tokenizing"
    )

    return dataset


def train():
    """Run LoRA fine-tuning with PEFT."""
    print("=" * 60)
    print("  Qwen 8B LoRA Fine-tuning (Optimized for A100)")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\nError: CUDA not available. Training requires a GPU.")
        sys.exit(1)

    # Check prerequisites
    if not check_data():
        sys.exit(1)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL}")
    print(f"Data: {DATA_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"LoRA rank: {LORA_R}")
    print(f"Learning rate: {LEARNING_RATE}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load and preprocess data
    print("Loading training data...")
    dataset = load_training_data(tokenizer)
    print(f"  Processed examples: {len(dataset)}")

    # Load base model in fp16 with optimizations
    print("\nLoading base model in fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",  # Scaled-dot-product attention for speed
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments (optimized for speed)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        fp16=True,
        optim="adamw_torch",  # PyTorch native optimizer (faster)
        lr_scheduler_type="linear",
        report_to="none",  # Disable wandb/tensorboard for speed
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
    )

    # Data collator
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Start training
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60 + "\n")

    try:
        trainer.train()

        # Save final adapters
        print("\nSaving adapters...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Adapters saved to: {OUTPUT_DIR}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Partial adapters may be saved in: {OUTPUT_DIR}")
        sys.exit(0)


if __name__ == "__main__":
    train()
