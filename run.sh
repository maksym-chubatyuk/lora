#!/bin/bash
# Full pipeline: setup, train, convert to GGUF, upload to GCS
# Usage: bash run.sh

set -e

BUCKET="gs://maksym-adapters"
MODEL_NAME="qwen3-vl-asuka"

echo "=== Qwen3-VL-8B-Thinking QLoRA Training ==="
echo ""

# Step 1: Virtual environment
if [ ! -d "venv" ]; then
    echo "[1/8] Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate

    echo "[2/8] Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install "transformers>=4.45.0" datasets peft accelerate
    pip install sentencepiece protobuf
    pip install flash-attn --no-build-isolation
    pip install gguf numpy
else
    echo "[1/8] Virtual environment exists, activating..."
    source venv/bin/activate
    echo "[2/8] Dependencies already installed"
fi

# Create output directory
mkdir -p output

echo ""
echo "[3/8] Converting training data to Qwen format..."
python3 convert_data.py -i asuka_training_data.jsonl -o asuka_training_data_qwen.jsonl

echo ""
echo "[4/8] Starting LoRA training..."
python3 train_qwen_vl.py

echo ""
echo "[5/8] Merging LoRA adapters into base model..."
python3 export_model.py --lora-path ./output/adapters --output ./output/merged --mode merge

echo ""
echo "[6/8] Setting up llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
    pip install -r llama.cpp/requirements.txt
fi

cd llama.cpp
if [ ! -f "build/bin/llama-quantize" ]; then
    echo "  Building llama-quantize..."
    cmake -B build
    cmake --build build --target llama-quantize -j
fi
cd ..

echo ""
echo "[7/8] Converting to GGUF and quantizing..."
# Convert to f16 GGUF
python3 llama.cpp/convert_hf_to_gguf.py output/merged --outfile output/model-f16.gguf --outtype f16

# Quantize to Q4_K_M
./llama.cpp/build/bin/llama-quantize output/model-f16.gguf output/model.gguf Q4_K_M

# Clean up f16 intermediate
rm -f output/model-f16.gguf

echo ""
echo "[8/8] Uploading to GCS..."
gsutil mb $BUCKET 2>/dev/null || true
gsutil cp output/model.gguf $BUCKET/${MODEL_NAME}.gguf

echo ""
echo "=== Complete! ==="
echo ""
echo "Output files:"
ls -lh output/*.gguf
echo ""
echo "Download with:"
echo "  gsutil cp $BUCKET/${MODEL_NAME}.gguf ."
