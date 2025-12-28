#!/usr/bin/env python3
"""
LoRA fine-tuning script for Qwen3-8B.
Uses dynamic padding for fast training.
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
    """Custom data collator with dynamic padding (pads to max in batch, not global max)."""

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # Find max length in batch
        max_length = max(len(f["input_ids"]) for f in features)

        # Round up to multiple for efficiency
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


# Configuration
MODEL = "Qwen/Qwen3-8B"
DATA_FILE = "asuka_training_data_qwen.jsonl"
OUTPUT_DIR = "output/adapters"

# Training hyperparameters
MAX_STEPS = 300
BATCH_SIZE = 8  # Increased - we have VRAM headroom
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 2  # Halved to keep effective batch same
WARMUP_STEPS = 20
LOGGING_STEPS = 10
SAVE_STEPS = 100
MAX_SEQ_LENGTH = 2048

# LoRA Configuration
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def train():
    """Run LoRA fine-tuning."""
    print("=" * 50)
    print("LoRA Fine-tuning")
    print("=" * 50)

    # Check CUDA
    if not torch.cuda.is_available():
        print("\nError: CUDA not available.")
        sys.exit(1)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL}")
    print(f"Data: {DATA_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"LoRA rank: {LORA_R}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load dataset
    print("Loading training data...")
    dataset = load_dataset("json", data_files={"train": DATA_FILE})

    def format_and_tokenize(example):
        """Format and tokenize in one step, NO padding here."""
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        result = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,  # NO PADDING - collator handles it
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(
        format_and_tokenize,
        remove_columns=["messages"],
        desc="Tokenizing"
    )
    print(f"  Training examples: {len(dataset['train'])}")

    # Load model
    print("\nLoading model in bf16 with Flash Attention 2...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # No gradient checkpointing = faster but more VRAM
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

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Compile for speed (PyTorch 2.0+)
    print("Compiling model with torch.compile...")
    model = torch.compile(model)

    # Training arguments
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
        bf16=True,  # Native on A100
        optim="adamw_torch_fused",  # Fused optimizer = faster
        lr_scheduler_type="linear",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,  # Avoid multiprocessing overhead
    )

    # Data collator with dynamic padding
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    # Train
    print("\n" + "-" * 50)
    print("Starting training...")
    print("-" * 50 + "\n")

    try:
        trainer.train()

        print("\nSaving adapters...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        print("\n" + "=" * 50)
        print("Training complete!")
        print(f"Adapters saved to: {OUTPUT_DIR}")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    train()
