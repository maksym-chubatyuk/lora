#!/usr/bin/env python3
"""
Export fine-tuned Qwen3-VL model for inference.

Supports:
1. Merge LoRA into base model (for further quantization)
2. Export to GGUF format (for llama.cpp/Ollama)
3. Keep as separate adapters (for transformers + PEFT)

Usage:
    # Merge LoRA adapters into base model
    python export_model.py --lora-path ./qwen3-vl-asuka-lora --output ./merged --mode merge

    # Export to GGUF (requires llama.cpp)
    python export_model.py --lora-path ./qwen3-vl-asuka-lora --output ./gguf --mode gguf

    # Just copy adapters for PEFT inference
    python export_model.py --lora-path ./qwen3-vl-asuka-lora --output ./inference --mode adapters
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel


MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"


def merge_lora(lora_path: str, output_path: str):
    """Merge LoRA adapters into base model."""
    print(f"Loading base model: {MODEL_ID}")

    # Load base model in fp16 for merging
    base_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapters from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)

    # Also save processor
    processor = AutoProcessor.from_pretrained(lora_path, trust_remote_code=True)
    processor.save_pretrained(output_path)

    print("Merge complete!")
    return output_path


def export_gguf(model_path: str, output_path: str, quant_type: str = "q4_k_m"):
    """
    Export to GGUF format for llama.cpp/Ollama.

    Note: This requires llama.cpp's convert scripts. The process is:
    1. Clone llama.cpp
    2. Run convert_hf_to_gguf.py
    3. Quantize with llama-quantize

    This function provides instructions since the conversion depends on
    external tools.
    """
    print("\n" + "="*60)
    print("GGUF Export Instructions")
    print("="*60)

    print(f"""
To convert to GGUF format for llama.cpp/Ollama:

1. Clone llama.cpp (if not already):
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp

2. Install dependencies:
   pip install -r requirements.txt

3. Convert to GGUF:
   python convert_hf_to_gguf.py {model_path} --outfile {output_path}/model-f16.gguf --outtype f16

4. Quantize (recommended for inference):
   ./llama-quantize {output_path}/model-f16.gguf {output_path}/model-{quant_type}.gguf {quant_type}

Recommended quantization types:
  - q4_k_m: Good balance of quality/size (recommended for RTX 3090)
  - q5_k_m: Higher quality, larger size
  - q8_0: Near-lossless, 8GB+ VRAM needed

For M4 Max (48GB unified memory):
  - q4_k_m or q5_k_m recommended
  - Can also use f16 if memory allows

NOTE: Qwen3-VL is a vision-language model. GGUF conversion may have
limited support for the vision encoder. Check llama.cpp releases for
the latest multimodal support.

Alternative: Use the model with transformers + PEFT directly.
""")

    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Write a helper script
    script_content = f"""#!/bin/bash
# GGUF conversion script for Qwen3-VL-Asuka

MODEL_PATH="{model_path}"
OUTPUT_DIR="{output_path}"
QUANT_TYPE="{quant_type}"

# Check if llama.cpp exists
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp
fi

cd llama.cpp

# Install requirements
pip install -r requirements.txt

# Convert to GGUF
echo "Converting to GGUF..."
python convert_hf_to_gguf.py "$MODEL_PATH" --outfile "$OUTPUT_DIR/model-f16.gguf" --outtype f16

# Build quantize tool if needed
if [ ! -f "llama-quantize" ]; then
    echo "Building llama.cpp..."
    make llama-quantize
fi

# Quantize
echo "Quantizing to $QUANT_TYPE..."
./llama-quantize "$OUTPUT_DIR/model-f16.gguf" "$OUTPUT_DIR/model-$QUANT_TYPE.gguf" $QUANT_TYPE

echo "Done! Output files:"
ls -lh "$OUTPUT_DIR"/*.gguf
"""

    script_path = Path(output_path) / "convert_to_gguf.sh"
    with open(script_path, "w") as f:
        f.write(script_content)

    print(f"Helper script written to: {script_path}")
    print("Run it with: bash " + str(script_path))


def copy_adapters(lora_path: str, output_path: str):
    """Copy LoRA adapters for PEFT inference."""
    print(f"Copying LoRA adapters from {lora_path} to {output_path}")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_dir = Path(lora_path)
    for file in lora_dir.iterdir():
        if file.is_file():
            shutil.copy2(file, output_dir / file.name)

    # Write inference script
    inference_script = f'''#!/usr/bin/env python3
"""
Inference script for fine-tuned Qwen3-VL-Asuka model.
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
LORA_PATH = "{output_path}"

def load_model(device_map="auto", load_in_4bit=True):
    """Load the fine-tuned model."""
    from transformers import BitsAndBytesConfig

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(model, LORA_PATH)
    processor = AutoProcessor.from_pretrained(LORA_PATH, trust_remote_code=True)

    return model, processor


def chat(model, processor, messages, max_new_tokens=512):
    """Generate a response."""
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    print("Loading model...")
    model, processor = load_model()

    # System prompt from training data
    system_prompt = """You are Asuka, a nineteen-year-old woman defined by fire, precision, and restless ambition..."""

    messages = [
        {{"role": "system", "content": system_prompt}},
        {{"role": "user", "content": "Hey, how are you today?"}}
    ]

    print("Generating response...")
    response = chat(model, processor, messages)
    print(response)
'''

    script_path = output_dir / "inference.py"
    with open(script_path, "w") as f:
        f.write(inference_script)

    print(f"Inference script written to: {script_path}")
    print("\nTo run inference:")
    print(f"  python {script_path}")


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned Qwen3-VL model")
    parser.add_argument(
        "--lora-path", "-l",
        type=str,
        required=True,
        help="Path to LoRA adapters"
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
        choices=["merge", "gguf", "adapters"],
        default="adapters",
        help="Export mode: merge (merge LoRA into base), gguf (convert to GGUF), adapters (copy for PEFT)"
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="q4_k_m",
        help="Quantization type for GGUF (default: q4_k_m)"
    )

    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.mode == "merge":
        merge_lora(args.lora_path, args.output)
    elif args.mode == "gguf":
        # For GGUF, we need to merge first
        print("Step 1: Merging LoRA adapters...")
        merged_path = str(Path(args.output) / "merged")
        merge_lora(args.lora_path, merged_path)

        print("\nStep 2: GGUF conversion...")
        export_gguf(merged_path, args.output, args.quant_type)
    elif args.mode == "adapters":
        copy_adapters(args.lora_path, args.output)

    print("\nExport complete!")


if __name__ == "__main__":
    main()
