#!/usr/bin/env python3
"""
LoRA fine-tuning script for Qwen3-8B using PyTorch on TPU.
Uses FSDP to shard model across all 4 TPU chips.

Usage:
    python train_qwen.py
"""

import os
import sys
from pathlib import Path
from functools import partial

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.fsdp as xla_fsdp
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm


# Configuration
MODEL_ID = "Qwen/Qwen3-8B"
DATA_FILE = "asuka_training_data_qwen.jsonl"
OUTPUT_DIR = "output/checkpoints"

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 1  # per device
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512
LOGGING_STEPS = 10
SAVE_STEPS = 100
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
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
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
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_fn(index):
    """Training function for each TPU core with FSDP."""
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()

    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Using FSDP to shard model across {world_size} TPU chips")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model on CPU first
    if rank == 0:
        print(f"Loading model: {MODEL_ID}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Configure LoRA
    if rank == 0:
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

    if rank == 0:
        model.print_trainable_parameters()

    # Wrap with FSDP - shard across all TPU chips
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        compute_dtype=torch.bfloat16,
        shard_param_on_dim_0=True,
        pin_layout_in_collective_ops=True,
    )

    if rank == 0:
        print("Model wrapped with FSDP")

    # Load dataset
    raw_dataset = load_dataset("json", data_files={"train": DATA_FILE})["train"]
    dataset = ChatDataset(raw_dataset, tokenizer, MAX_SEQ_LENGTH)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )

    if rank == 0:
        print(f"Training examples: {len(dataset)}")
        print(f"Batch size per device: {BATCH_SIZE}")
        print(f"Total batch size: {BATCH_SIZE * world_size * GRADIENT_ACCUMULATION}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Training loop
    model.train()
    global_step = 0
    total_loss = 0.0

    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n[Epoch {epoch + 1}/{NUM_EPOCHS}]")
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        else:
            progress_bar = dataloader

        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
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

                if global_step % LOGGING_STEPS == 0 and rank == 0:
                    avg_loss = total_loss / (LOGGING_STEPS * GRADIENT_ACCUMULATION)
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})
                    total_loss = 0.0

                if global_step % SAVE_STEPS == 0 and rank == 0:
                    save_path = Path(OUTPUT_DIR) / f"step_{global_step}"
                    save_path.mkdir(parents=True, exist_ok=True)
                    # Save consolidated state dict
                    xm.save(model.state_dict(), save_path / "model.pt")
                    tokenizer.save_pretrained(save_path)
                    print(f"\nSaved checkpoint to {save_path}")

    # Final save
    if rank == 0:
        final_path = Path(OUTPUT_DIR) / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        xm.save(model.state_dict(), final_path / "model.pt")
        tokenizer.save_pretrained(final_path)
        print(f"\nTraining complete! Model saved to {final_path}")


def main():
    print("=" * 60)
    print("PyTorch + TPU XLA FSDP LoRA Fine-tuning")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nModel: {MODEL_ID}")
    print(f"LoRA rank: {LORA_R}")
    print("Launching FSDP training across all TPU chips...")
    print()

    xmp.spawn(train_fn, args=(), nprocs=None)


if __name__ == "__main__":
    main()
