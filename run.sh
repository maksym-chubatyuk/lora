#!/bin/bash
# Full pipeline: setup TPU environment, train with JAX/EasyDeL, export to GGUF
# Usage: bash run.sh

set -e

BUCKET="gs://maksym-adapters"
MODEL_NAME="qwen3-asuka"

echo "=== Qwen3-8B LoRA Training on TPU (PyTorch XLA) ==="
echo ""

# Step 1: Install system dependencies (Ubuntu)
echo "[1/8] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.10-venv python3-pip cmake build-essential

# Step 2: Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "[2/8] Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate

    pip install --upgrade pip

    echo "  Installing PyTorch for TPU..."
    pip install torch~=2.6.0 torch_xla[tpu]~=2.6.0 \
        -f https://storage.googleapis.com/libtpu-releases/index.html

    echo "  Installing transformers..."
    pip install transformers

    echo "  Installing PEFT..."
    pip install peft

    echo "  Installing accelerate..."
    pip install accelerate

    echo "  Installing datasets..."
    pip install datasets

    echo "  Installing utilities..."
    pip install numpy tqdm gguf
else
    echo ""
    echo "[2/8] Virtual environment exists, activating..."
    source venv/bin/activate
fi

# Verify PyTorch XLA sees TPUs
echo ""
echo "[4/8] Checking TPU devices..."
python3 -c "import torch_xla.core.xla_model as xm; print(f'TPU devices: {xm.get_xla_supported_devices()}')"

# Create output directory
mkdir -p output

echo ""
echo "[5/8] Converting training data to Qwen format..."
python3 convert_data.py -i asuka_training_data.jsonl -o asuka_training_data_qwen.jsonl

echo ""
echo "[6/8] Starting LoRA training on TPU..."
python3 train_qwen.py

echo ""
echo "[7/8] Exporting model to HuggingFace format and GGUF..."

# Setup llama.cpp
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp
    pip install -q -r llama.cpp/requirements.txt
fi

cd llama.cpp
if [ ! -f "build/bin/llama-quantize" ]; then
    echo "  Building llama-quantize..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target llama-quantize -j$(nproc)
fi
cd ..

# Export and convert
python3 export_model.py --checkpoint ./output/checkpoints --output ./output/merged --mode merge

# Convert to GGUF
python3 llama.cpp/convert_hf_to_gguf.py output/merged --outfile output/model-f16.gguf --outtype f16

# Quantize to Q4_K_M
./llama.cpp/build/bin/llama-quantize output/model-f16.gguf output/model.gguf Q4_K_M

# Clean up
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
echo ""
echo "Run locally with Ollama:"
echo "  ollama create asuka -f Modelfile"
echo "  ollama run asuka"
