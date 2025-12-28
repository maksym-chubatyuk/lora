#!/usr/bin/env python3
"""
LoRA fine-tuning script for Qwen3-8B using PyTorch on TPU.
Uses torch_xla for TPU support and PEFT for LoRA.

Usage:
    python train_qwen.py
"""

import os
import sys
from pathlib import Path

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from transformers import AutoModelForCausalLM, AutoTokenizer
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
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
MAX_SEQ_LENGTH = 1024
LOGGING_STEPS = 10
SAVE_STEPS = 200
GRADIENT_ACCUMULATION = 4

# LoRA configuration
LORA_R = 64
LORA_ALPHA = 128
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

        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_fn(index):
    """Training function for each TPU core."""
    device = xm.xla_device()
    print(f"[Core {index}] Using device: {device}")

    # Load tokenizer
    print(f"[Core {index}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    print(f"[Core {index}] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Configure LoRA
    print(f"[Core {index}] Configuring LoRA...")
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

    # Move to TPU
    model = model.to(device)

    # Load dataset
    print(f"[Core {index}] Loading dataset...")
    raw_dataset = load_dataset("json", data_files={"train": DATA_FILE})["train"]

    dataset = ChatDataset(raw_dataset, tokenizer, MAX_SEQ_LENGTH)

    # Create sampler for distributed training
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )

    # Wrap with parallel loader for TPU
    para_loader = pl.ParallelLoader(dataloader, [device])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # Learning rate scheduler
    num_training_steps = len(dataloader) * NUM_EPOCHS // GRADIENT_ACCUMULATION
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
        eta_min=LEARNING_RATE * 0.1,
    )

    # Training loop
    model.train()
    global_step = 0
    total_loss = 0.0

    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        per_device_loader = para_loader.per_device_loader(device)

        if index == 0:
            print(f"\n[Epoch {epoch + 1}/{NUM_EPOCHS}]")

        progress_bar = tqdm(
            per_device_loader,
            desc=f"Epoch {epoch + 1}",
            disable=(index != 0),
        )

        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

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
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % LOGGING_STEPS == 0 and index == 0:
                    avg_loss = total_loss / LOGGING_STEPS
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    total_loss = 0.0

                # Save checkpoint
                if global_step % SAVE_STEPS == 0 and index == 0:
                    save_path = Path(OUTPUT_DIR) / f"step_{global_step}"
                    save_path.mkdir(parents=True, exist_ok=True)

                    # Save only on main process
                    xm.save(model.state_dict(), save_path / "adapter_model.bin")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"\nSaved checkpoint to {save_path}")

    # Final save
    if index == 0:
        final_path = Path(OUTPUT_DIR) / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        xm.save(model.state_dict(), final_path / "adapter_model.bin")
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"\nTraining complete! Model saved to {final_path}")


def main():
    print("=" * 60)
    print("PyTorch + TPU XLA LoRA Fine-tuning")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get number of TPU cores
    num_cores = 4  # Your 4 TPU setup

    print(f"\nStarting training on {num_cores} TPU cores...")
    print(f"Model: {MODEL_ID}")
    print(f"LoRA rank: {LORA_R}")
    print(f"Batch size per core: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"Effective batch size: {BATCH_SIZE * num_cores * GRADIENT_ACCUMULATION}")
    print()

    # Launch distributed training
    xmp.spawn(train_fn, args=(), nprocs=num_cores)


if __name__ == "__main__":
    main()
