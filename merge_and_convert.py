#!/usr/bin/env python3
"""
Merge LoRA adapters into base model and convert to GGUF format.
Run this after training to create a standalone fp16 GGUF model.

Output: output/model-f16.gguf (~16GB)
"""

import os
import sys
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

# =============================================================================
# Configuration
# =============================================================================

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
ADAPTER_PATH = "output/adapters"
MERGED_PATH = "output/merged_bf16"
GGUF_PATH = "output/model-bf16.gguf"
LLAMA_CPP_PATH = "llama.cpp"


def check_prerequisites():
    """Check that all prerequisites are met."""
    # Check for adapters
    adapter_path = Path(ADAPTER_PATH)
    if not adapter_path.exists():
        print(f"Error: Adapter path {ADAPTER_PATH} does not exist!")
        print("Run training first: python train.py")
        return False

    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.exists():
        print(f"Error: No adapter_config.json found in {ADAPTER_PATH}")
        return False

    # Check for llama.cpp
    llama_cpp = Path(LLAMA_CPP_PATH)
    if not llama_cpp.exists():
        print(f"Error: llama.cpp not found at {LLAMA_CPP_PATH}")
        print("Clone it with: git clone https://github.com/ggerganov/llama.cpp")
        return False

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"Error: convert_hf_to_gguf.py not found in {LLAMA_CPP_PATH}")
        print("Make sure you have the latest llama.cpp")
        return False

    return True


def merge_adapters():
    """Merge LoRA adapters into the base model."""
    print("=" * 60)
    print("Step 1: Merging LoRA Adapters (bf16)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Warning: CUDA not available. This will be slow on CPU.")

    print(f"\nLoading base model in bf16: {MODEL}")
    print("  This may take a few minutes...")

    base_model = AutoModelForVision2Seq.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"\nLoading adapters from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("\nMerging adapters into base model...")
    model = model.merge_and_unload()

    # Create output directory
    os.makedirs(MERGED_PATH, exist_ok=True)

    print(f"\nSaving merged model to: {MERGED_PATH}")
    model.save_pretrained(MERGED_PATH)

    # Also save processor
    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    processor.save_pretrained(MERGED_PATH)

    print("\nMerge complete!")

    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory used: {memory_gb:.2f} GB")


def convert_to_gguf():
    """Convert merged HuggingFace model to GGUF format (bf16)."""
    print("\n" + "=" * 60)
    print("Step 2: Converting to GGUF (bf16)")
    print("=" * 60)

    convert_script = Path(LLAMA_CPP_PATH) / "convert_hf_to_gguf.py"

    # Create output directory
    output_dir = Path(GGUF_PATH).parent
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nInput: {MERGED_PATH}")
    print(f"Output: {GGUF_PATH}")
    print("\nRunning conversion to bf16 GGUF...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(convert_script),
                MERGED_PATH,
                "--outfile", GGUF_PATH,
                "--outtype", "bf16",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        return False

    # Check output exists
    if not Path(GGUF_PATH).exists():
        print(f"Error: GGUF not created at {GGUF_PATH}")
        return False

    # Print file size
    gguf_file = Path(GGUF_PATH)
    size_gb = gguf_file.stat().st_size / (1024**3)
    print(f"\nGGUF model created successfully!")
    print(f"  File: {GGUF_PATH}")
    print(f"  Size: {size_gb:.2f} GB")

    return True


def cleanup_intermediate():
    """Info about cleaning up intermediate files."""
    merged_path = Path(MERGED_PATH)
    if merged_path.exists():
        size_gb = sum(f.stat().st_size for f in merged_path.rglob('*') if f.is_file()) / (1024**3)
        print(f"\nIntermediate files at {MERGED_PATH}: {size_gb:.2f} GB")
        print("You can delete this folder to save space:")
        print(f"  rm -rf {MERGED_PATH}")


def main():
    print("=" * 60)
    print("  LoRA Merge & GGUF Conversion (fp16)")
    print("=" * 60)
    print()

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Step 1: Merge adapters
    merge_adapters()

    # Step 2: Convert to GGUF
    success = convert_to_gguf()

    if success:
        print("\n" + "=" * 60)
        print("  Conversion Complete!")
        print("=" * 60)
        print(f"\nYour GGUF file is ready: {GGUF_PATH}")
        print("\nTo quantize locally (after downloading):")
        print("  ./llama.cpp/build/bin/llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M")
        cleanup_intermediate()
    else:
        print("\nConversion failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
