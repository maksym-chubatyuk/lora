#!/usr/bin/env python3
"""
Export JAX-trained Qwen3-8B LoRA model for inference.

Converts JAX/EasyDeL checkpoints to formats usable for inference:
1. PyTorch/HuggingFace format (merged weights)
2. GGUF format (for llama.cpp/Ollama)

Usage:
    # Convert JAX checkpoint to merged HuggingFace format
    python export_model.py --checkpoint ./output/checkpoints --output ./output/merged --mode merge

    # Convert to GGUF (includes merge step)
    python export_model.py --checkpoint ./output/checkpoints --output ./output/gguf --mode gguf
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Only import JAX if we're doing the merge
# PyTorch is imported for the GGUF conversion step


MODEL_ID = "Qwen/Qwen3-8B"


def jax_to_pytorch_state_dict(jax_params: dict, model_config: dict) -> dict:
    """
    Convert JAX parameter tree to PyTorch state dict.

    This handles the different naming conventions and array formats
    between JAX/Flax and PyTorch.
    """
    import torch

    state_dict = {}

    def flatten_params(params, prefix=""):
        """Recursively flatten nested JAX params."""
        for key, value in params.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flatten_params(value, full_key)
            else:
                # Convert to PyTorch tensor
                # JAX uses (out, in) for linear layers, PyTorch uses (in, out)
                arr = np.array(value)

                # Handle transposition for linear layers
                if "kernel" in key and arr.ndim == 2:
                    arr = arr.T
                    full_key = full_key.replace("kernel", "weight")
                elif "scale" in key:
                    full_key = full_key.replace("scale", "weight")

                # Convert naming: Flax uses underscores, PyTorch uses dots
                full_key = full_key.replace("_", ".")

                state_dict[full_key] = torch.from_numpy(arr)

    flatten_params(jax_params)
    return state_dict


def load_jax_checkpoint(checkpoint_path: str):
    """Load JAX checkpoint using Orbax."""
    import orbax.checkpoint as ocp

    print(f"Loading JAX checkpoint from: {checkpoint_path}")

    checkpointer = ocp.StandardCheckpointer()

    # Find the latest checkpoint
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    # EasyDeL saves checkpoints in subdirectories with step numbers
    checkpoint_dirs = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name)
    )

    if checkpoint_dirs:
        latest = checkpoint_dirs[-1]
        print(f"Found checkpoint at step: {latest.name}")
    else:
        latest = checkpoint_dir

    # Load the checkpoint
    restored = checkpointer.restore(latest)

    return restored


def merge_and_export_hf(checkpoint_path: str, output_path: str):
    """
    Load JAX checkpoint, merge LoRA weights, and export to HuggingFace format.
    """
    import jax
    import jax.numpy as jnp
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print("=" * 60)
    print("Exporting JAX checkpoint to HuggingFace format")
    print("=" * 60)

    # Load JAX checkpoint
    checkpoint = load_jax_checkpoint(checkpoint_path)

    # The checkpoint should contain:
    # - params: the LoRA-adapted parameters
    # - or separate base_params + lora_params

    if "params" in checkpoint:
        params = checkpoint["params"]
    else:
        print("Warning: Unexpected checkpoint format. Trying to find parameters...")
        params = checkpoint

    print(f"Loaded parameters with {len(params)} top-level keys")

    # Load base model in PyTorch
    print(f"\nLoading base model: {MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",  # Keep on CPU for merging
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Convert JAX params to PyTorch state dict
    print("Converting JAX parameters to PyTorch format...")
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    jax_state_dict = jax_to_pytorch_state_dict(params, config.to_dict())

    # Update base model state dict with fine-tuned weights
    base_state_dict = base_model.state_dict()

    updated_count = 0
    for key, value in jax_state_dict.items():
        if key in base_state_dict:
            if base_state_dict[key].shape == value.shape:
                base_state_dict[key] = value
                updated_count += 1
            else:
                print(f"  Shape mismatch for {key}: {base_state_dict[key].shape} vs {value.shape}")

    print(f"Updated {updated_count} parameters")

    # Load updated weights
    base_model.load_state_dict(base_state_dict)

    # Save
    print(f"\nSaving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    base_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("Export complete!")
    return output_path


def export_gguf(model_path: str, output_path: str, quant_type: str = "Q4_K_M"):
    """
    Convert HuggingFace model to GGUF format.

    This requires llama.cpp to be available.
    """
    import subprocess

    print("\n" + "=" * 60)
    print("Converting to GGUF format")
    print("=" * 60)

    os.makedirs(output_path, exist_ok=True)

    # Check for llama.cpp
    llama_cpp_path = Path("llama.cpp")
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print(f"""
llama.cpp not found. Please set it up first:

    git clone https://github.com/ggerganov/llama.cpp
    pip install -r llama.cpp/requirements.txt

Then run this script again.
""")

        # Write helper script
        helper_script = f"""#!/bin/bash
# GGUF conversion helper script

set -e

MODEL_PATH="{model_path}"
OUTPUT_PATH="{output_path}"
QUANT_TYPE="{quant_type}"

# Clone llama.cpp if needed
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi

# Install requirements
pip install -r llama.cpp/requirements.txt

# Build quantize tool
cd llama.cpp
cmake -B build
cmake --build build --target llama-quantize -j
cd ..

# Convert to GGUF
python llama.cpp/convert_hf_to_gguf.py "$MODEL_PATH" \\
    --outfile "$OUTPUT_PATH/model-f16.gguf" \\
    --outtype f16

# Quantize
./llama.cpp/build/bin/llama-quantize \\
    "$OUTPUT_PATH/model-f16.gguf" \\
    "$OUTPUT_PATH/model-$QUANT_TYPE.gguf" \\
    $QUANT_TYPE

# Clean up f16 intermediate
rm -f "$OUTPUT_PATH/model-f16.gguf"

echo ""
echo "Done! GGUF file:"
ls -lh "$OUTPUT_PATH"/*.gguf
"""
        helper_path = Path(output_path) / "convert_to_gguf.sh"
        with open(helper_path, "w") as f:
            f.write(helper_script)

        print(f"Helper script written to: {helper_path}")
        print(f"Run: bash {helper_path}")
        return

    # Run conversion
    print("Converting to f16 GGUF...")
    f16_path = Path(output_path) / "model-f16.gguf"

    subprocess.run([
        sys.executable, str(convert_script),
        model_path,
        "--outfile", str(f16_path),
        "--outtype", "f16"
    ], check=True)

    # Quantize
    print(f"Quantizing to {quant_type}...")
    quantize_bin = llama_cpp_path / "build" / "bin" / "llama-quantize"

    if not quantize_bin.exists():
        print("Building llama-quantize...")
        subprocess.run(["cmake", "-B", "build"], cwd=llama_cpp_path, check=True)
        subprocess.run(["cmake", "--build", "build", "--target", "llama-quantize", "-j"],
                       cwd=llama_cpp_path, check=True)

    final_path = Path(output_path) / f"model-{quant_type}.gguf"
    subprocess.run([
        str(quantize_bin),
        str(f16_path),
        str(final_path),
        quant_type
    ], check=True)

    # Clean up
    f16_path.unlink()

    print(f"\nGGUF file created: {final_path}")
    print(f"Size: {final_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Export JAX-trained model for inference")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to JAX/EasyDeL checkpoint directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["merge", "gguf"],
        default="gguf",
        help="Export mode: merge (HuggingFace format), gguf (llama.cpp format)"
    )
    parser.add_argument(
        "--quant-type", "-q",
        type=str,
        default="Q4_K_M",
        help="GGUF quantization type (default: Q4_K_M)"
    )

    args = parser.parse_args()

    if args.mode == "merge":
        merge_and_export_hf(args.checkpoint, args.output)
    elif args.mode == "gguf":
        # First merge to HF format
        merged_path = str(Path(args.output) / "merged")
        merge_and_export_hf(args.checkpoint, merged_path)

        # Then convert to GGUF
        export_gguf(merged_path, args.output, args.quant_type)

    print("\nExport complete!")


if __name__ == "__main__":
    main()
