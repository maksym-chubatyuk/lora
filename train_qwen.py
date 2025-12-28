#!/usr/bin/env python3
"""
LoRA fine-tuning script for Qwen3-8B using PyTorch on TPU.
Single-process version that works with limited TPU memory.

Usage:
    python train_qwen.py
"""

import os
import sys
from pathlib import Path

import torch
import torch_xla.core.xla_model as xm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


# Configuration
MODEL_ID = "Qwen/Qwen3-8B"
DATA_FILE = "asuka_training_data_qwen.jsonl"
OUTPUT_DIR = "output/checkpoints"

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 1  # Small batch for memory
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512  # Reduced for memory
LOGGING_STEPS = 10
SAVE_STEPS = 100
GRADIENT_ACCUMULATION = 16  # Accumulate more to compensate small batch

# LoRA configuration
LORA_R = 32  # Reduced rank for memory
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class ChatDataset(torch.utils.data.Dataset):
    """Dataset for chat-formatted training data."""

    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    print("=" * 60)
    print("PyTorch + TPU XLA LoRA Fine-tuning")
    print("=" * 60)

    # Get TPU device
    device = xm.xla_device()
    print(f"\nUsing device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model in lower precision
    print(f"Loading model: {MODEL_ID}")
    print("This may take a few minutes...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Configure LoRA
    print("Configuring LoRA...")
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

    # Note: gradient checkpointing doesn't work with TPU XLA

    # Move to TPU
    print("Moving model to TPU...")
    model = model.to(device)

    # Load dataset
    print(f"Loading dataset from {DATA_FILE}...")
    raw_dataset = load_dataset("json", data_files={"train": DATA_FILE})["train"]
    dataset = ChatDataset(raw_dataset, tokenizer, MAX_SEQ_LENGTH)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    print(f"Training examples: {len(dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # Training loop
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60 + "\n")

    model.train()
    global_step = 0
    total_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(NUM_EPOCHS):
        print(f"\n[Epoch {epoch + 1}/{NUM_EPOCHS}]")

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            # Move batch to TPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / GRADIENT_ACCUMULATION
            loss.backward()

            total_loss += loss.item() * GRADIENT_ACCUMULATION

            if (step + 1) % GRADIENT_ACCUMULATION == 0:
                xm.optimizer_step(optimizer)
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % LOGGING_STEPS == 0:
                    avg_loss = total_loss / (LOGGING_STEPS * GRADIENT_ACCUMULATION)
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})
                    total_loss = 0.0

                # Save checkpoint
                if global_step % SAVE_STEPS == 0:
                    save_path = Path(OUTPUT_DIR) / f"step_{global_step}"
                    save_path.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"\nSaved checkpoint to {save_path}")

    # Final save
    final_path = Path(OUTPUT_DIR) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
