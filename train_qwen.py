#!/usr/bin/env python3
"""
LoRA fine-tuning script for Qwen3-8B using JAX/EasyDeL on TPU.

Usage:
    python train_qwen.py
"""

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as PS
from jax.experimental.mesh_utils import create_device_mesh

from datasets import load_dataset
from transformers import AutoTokenizer

from easydel import (
    AutoEasyDeLModelForCausalLM,
    EasyDeLXRapTureConfig,
    TrainArguments,
    SFTTrainer,
)


# Configuration
MODEL_ID = "Qwen/Qwen3-8B"
DATA_FILE = "asuka_training_data_qwen.jsonl"
OUTPUT_DIR = "output/checkpoints"

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 8  # per device
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.03
MAX_SEQ_LENGTH = 2048
LOGGING_STEPS = 10
SAVE_STEPS = 100

# LoRA configuration
LORA_DIM = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def setup_tpu_mesh():
    """Setup TPU mesh for distributed training."""
    devices = jax.devices()
    num_devices = len(devices)

    print(f"Found {num_devices} TPU device(s)")

    if num_devices == 1:
        mesh = Mesh(jax.devices(), ("dp",))
    elif num_devices == 4:
        # 2x2 mesh: data parallel x model parallel
        device_mesh = create_device_mesh((2, 2))
        mesh = Mesh(device_mesh, ("dp", "mp"))
    elif num_devices == 8:
        # 2x4 or 4x2 mesh
        device_mesh = create_device_mesh((2, 4))
        mesh = Mesh(device_mesh, ("dp", "mp"))
    else:
        # Default: all data parallel
        mesh = Mesh(jax.devices(), ("dp",))

    return mesh


def load_and_prepare_data(tokenizer, max_length: int):
    """Load and tokenize the training data."""
    print(f"Loading data from {DATA_FILE}...")

    dataset = load_dataset("json", data_files={"train": DATA_FILE})

    def tokenize_function(examples):
        """Tokenize and format for training."""
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="np",
        )

        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["messages"],
        desc="Tokenizing"
    )

    print(f"Training examples: {len(tokenized_dataset['train'])}")
    return tokenized_dataset["train"]


def train():
    """Run LoRA fine-tuning on TPU."""
    print("=" * 50)
    print("JAX/EasyDeL LoRA Fine-tuning on TPU")
    print("=" * 50)

    # Check JAX devices
    print(f"\nJAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    if jax.default_backend() != "tpu":
        print("\nWarning: Not running on TPU. Performance will be limited.")

    # Setup mesh
    mesh = setup_tpu_mesh()
    print(f"Mesh shape: {mesh.shape}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    print(f"Loading model: {MODEL_ID}")
    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        sharding_axis_dims=(1, -1, 1, 1),  # batch, sequence, hidden, heads
        sharding_axis_names=("dp", "fsdp", "tp", "sp"),
        trust_remote_code=True,
    )

    print("Model loaded successfully!")

    # Configure LoRA
    print(f"\nConfiguring LoRA (rank={LORA_DIM}, alpha={LORA_ALPHA})")
    rapture_config = EasyDeLXRapTureConfig(
        parameters=params,
        lora_dim=LORA_DIM,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        fully_fine_tune_parameters=[],  # No layers fully fine-tuned
        lora_fine_tune_parameters=LORA_TARGET_MODULES,
        verbose=True,
    )

    # Load data
    train_dataset = load_and_prepare_data(tokenizer, MAX_SEQ_LENGTH)

    # Training arguments
    total_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION * len(jax.devices()))) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    print(f"\nTotal training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    train_args = TrainArguments(
        model_name="qwen3-asuka-lora",
        num_train_epochs=NUM_EPOCHS,
        total_batch_size=BATCH_SIZE * len(jax.devices()) * GRADIENT_ACCUMULATION,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        learning_rate_end=LEARNING_RATE * 0.1,
        warmup_steps=warmup_steps,
        optimizer="adamw",
        scheduler="linear",
        weight_decay=0.01,
        max_sequence_length=MAX_SEQ_LENGTH,
        gradient_checkpointing="nothing_saveable",
        sharding_array=(1, -1, 1, 1),
        use_pjit_attention_force=False,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        step_start_point=0,
        training_time=None,
        do_train=True,
        do_eval=False,
        track_memory=True,
        save_steps=SAVE_STEPS,
        save_dir=OUTPUT_DIR,
        save_total_limit=3,
        logging_steps=LOGGING_STEPS,
        use_wandb=False,
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        arguments=train_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        dataset_text_field=None,  # Already tokenized
        rapture_config=rapture_config,
        formatting_func=None,
    )

    # Train
    print("\n" + "-" * 50)
    print("Starting training...")
    print("-" * 50 + "\n")

    try:
        output = trainer.train()

        print("\n" + "=" * 50)
        print("Training complete!")
        print(f"Checkpoints saved to: {OUTPUT_DIR}")
        print("=" * 50)

        return output

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    train()
