#!/usr/bin/env python3
"""
LoRA fine-tuning script for Qwen3-8B using JAX on TPU.
Uses lorax for LoRA and a simple training loop.

Usage:
    python train_qwen.py
"""

import os
import sys
import json
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as PS, NamedSharding
from jax.experimental.mesh_utils import create_device_mesh
import optax
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, AutoConfig
from datasets import load_dataset


# Configuration
MODEL_ID = "Qwen/Qwen2.5-7B"  # Using Qwen2.5 which has Flax support
DATA_FILE = "asuka_training_data_qwen.jsonl"
OUTPUT_DIR = "output/checkpoints"

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 4  # per device
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
MAX_SEQ_LENGTH = 1024
LOGGING_STEPS = 10
SAVE_STEPS = 200

# LoRA configuration
LORA_RANK = 64
LORA_ALPHA = 128


def create_mesh():
    """Create TPU mesh for distributed training."""
    devices = jax.devices()
    num_devices = len(devices)
    print(f"Found {num_devices} TPU device(s)")

    if num_devices == 4:
        mesh = Mesh(create_device_mesh((2, 2)), ("dp", "mp"))
    else:
        mesh = Mesh(np.array(devices), ("dp",))

    return mesh


def load_and_prepare_data(tokenizer, max_length: int):
    """Load and tokenize the training data."""
    print(f"Loading data from {DATA_FILE}...")

    dataset = load_dataset("json", data_files={"train": DATA_FILE})

    def tokenize_function(examples):
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

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["messages"],
        desc="Tokenizing"
    )

    print(f"Training examples: {len(tokenized['train'])}")
    return tokenized["train"]


def create_lora_params(params, rank: int, alpha: int, target_modules: list):
    """Initialize LoRA parameters for target modules."""
    lora_params = {}
    scale = alpha / rank

    def init_lora_for_layer(path, param):
        """Initialize LoRA A and B matrices for a parameter."""
        if param.ndim != 2:
            return None

        # Check if this is a target module
        path_str = ".".join(path)
        is_target = any(target in path_str for target in target_modules)
        if not is_target:
            return None

        in_features, out_features = param.shape

        # Initialize A with small random values, B with zeros
        key = jax.random.PRNGKey(hash(path_str) % 2**32)
        lora_a = jax.random.normal(key, (in_features, rank)) * 0.01
        lora_b = jnp.zeros((rank, out_features))

        return {"A": lora_a, "B": lora_b, "scale": scale}

    def traverse(params, path=[]):
        result = {}
        for key, value in params.items():
            current_path = path + [key]
            if isinstance(value, dict):
                sub_result = traverse(value, current_path)
                if sub_result:
                    result[key] = sub_result
            else:
                lora = init_lora_for_layer(current_path, value)
                if lora is not None:
                    result[key] = lora
        return result

    return traverse(params)


def apply_lora(params, lora_params, x, path=[]):
    """Apply LoRA to forward pass."""
    # This is a simplified version - in practice you'd integrate this
    # into the model's forward pass
    pass


@partial(jax.jit, static_argnums=(3,))
def train_step(params, lora_params, opt_state, apply_fn, batch, optimizer):
    """Single training step."""

    def loss_fn(lora_p):
        # Merge LoRA into params for forward pass
        merged = merge_lora_params(params, lora_p)

        logits = apply_fn(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            params=merged,
            train=True,
        ).logits

        # Shift for causal LM loss
        shift_logits = logits[..., :-1, :]
        shift_labels = batch["labels"][..., 1:]

        # Cross entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels
        )

        # Mask padding
        mask = (shift_labels != -100) & (shift_labels != 0)
        loss = jnp.sum(loss * mask) / jnp.sum(mask)

        return loss

    loss, grads = jax.value_and_grad(loss_fn)(lora_params)
    updates, opt_state = optimizer.update(grads, opt_state, lora_params)
    lora_params = optax.apply_updates(lora_params, updates)

    return lora_params, opt_state, loss


def merge_lora_params(base_params, lora_params):
    """Merge LoRA parameters into base parameters."""
    def merge(base, lora):
        if isinstance(lora, dict) and "A" in lora:
            # This is a LoRA layer
            delta = jnp.matmul(lora["A"], lora["B"]) * lora["scale"]
            return base + delta
        elif isinstance(base, dict):
            return {k: merge(base[k], lora.get(k, {})) for k in base}
        else:
            return base

    return merge(base_params, lora_params)


def save_checkpoint(lora_params, step, output_dir):
    """Save LoRA parameters."""
    checkpoint_dir = Path(output_dir) / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Flatten and save as numpy
    flat_params = jax.tree_util.tree_leaves(lora_params)
    flat_structure = jax.tree_util.tree_structure(lora_params)

    np.savez(
        checkpoint_dir / "lora_params.npz",
        *flat_params
    )

    # Save structure info
    with open(checkpoint_dir / "structure.json", "w") as f:
        json.dump({"step": step}, f)

    print(f"Saved checkpoint to {checkpoint_dir}")


def train():
    """Run LoRA fine-tuning on TPU."""
    print("=" * 60)
    print("JAX LoRA Fine-tuning on TPU")
    print("=" * 60)

    print(f"\nJAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    # Setup mesh
    mesh = create_mesh()
    print(f"Mesh: {mesh}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {MODEL_ID}")
    print("This may take a few minutes...")

    model = FlaxAutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,
        trust_remote_code=True,
    )

    params = model.params
    print("Model loaded!")

    # Initialize LoRA
    print(f"\nInitializing LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_params = create_lora_params(params, LORA_RANK, LORA_ALPHA, target_modules)

    num_lora_params = sum(p.size for p in jax.tree_util.tree_leaves(lora_params))
    print(f"LoRA parameters: {num_lora_params:,}")

    # Load data
    train_dataset = load_and_prepare_data(tokenizer, MAX_SEQ_LENGTH)

    # Create optimizer
    num_steps = (len(train_dataset) // (BATCH_SIZE * len(jax.devices()))) * NUM_EPOCHS

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=num_steps,
        end_value=LEARNING_RATE * 0.1,
    )

    optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.01)
    opt_state = optimizer.init(lora_params)

    print(f"\nTotal steps: {num_steps}")
    print(f"Warmup steps: {WARMUP_STEPS}")

    # Training loop
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60 + "\n")

    global_step = 0
    total_loss = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Shuffle dataset
        train_dataset = train_dataset.shuffle(seed=epoch)

        # Create batches
        num_batches = len(train_dataset) // (BATCH_SIZE * len(jax.devices()))

        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}")

        for batch_idx in progress_bar:
            # Get batch
            start_idx = batch_idx * BATCH_SIZE * len(jax.devices())
            end_idx = start_idx + BATCH_SIZE * len(jax.devices())

            batch = {
                "input_ids": jnp.array(train_dataset[start_idx:end_idx]["input_ids"]),
                "attention_mask": jnp.array(train_dataset[start_idx:end_idx]["attention_mask"]),
                "labels": jnp.array(train_dataset[start_idx:end_idx]["labels"]),
            }

            # Train step
            lora_params, opt_state, loss = train_step(
                params, lora_params, opt_state, model.__call__, batch, optimizer
            )

            total_loss += float(loss)
            global_step += 1

            # Logging
            if global_step % LOGGING_STEPS == 0:
                avg_loss = total_loss / LOGGING_STEPS
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                total_loss = 0.0

            # Save checkpoint
            if global_step % SAVE_STEPS == 0:
                save_checkpoint(lora_params, global_step, OUTPUT_DIR)

    # Final save
    save_checkpoint(lora_params, global_step, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    train()
