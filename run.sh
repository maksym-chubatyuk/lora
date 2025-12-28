#!/bin/bash
# Full pipeline: setup TPU environment, train with JAX/EasyDeL, export to GGUF
# Usage: bash run.sh

set -e

BUCKET="gs://maksym-adapters"
MODEL_NAME="qwen3-asuka"

echo "=== Qwen3-8B LoRA Training on TPU (JAX/EasyDeL) ==="
echo ""

# Step 1: Check TPU
echo "[1/7] Checking TPU devices..."
python3 -c "import jax; print(f'JAX backend: {jax.default_backend()}'); print(f'Devices: {jax.devices()}')" || {
    echo "JAX not installed or TPU not available"
    echo "Installing JAX for TPU..."
    pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
}

# Step 2: Virtual environment and dependencies
if [ ! -d "venv" ]; then
    echo ""
    echo "[2/7] Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate

    echo "Installing dependencies..."
    pip install --upgrade pip

    # JAX for TPU
    pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

    # EasyDeL and JAX ecosystem
    pip install easydel optax flax orbax-checkpoint

    # Data loading
    pip install datasets grain transformers

    # Utilities
    pip install numpy tqdm
else
    echo ""
    echo "[2/7] Virtual environment exists, activating..."
    source venv/bin/activate
fi

# Create output directory
mkdir -p output

echo ""
echo "[3/7] Converting training data to Qwen format..."
python3 convert_data.py -i asuka_training_data.jsonl -o asuka_training_data_qwen.jsonl

echo ""
echo "[4/7] Starting LoRA training on TPU..."
python3 train_qwen.py

echo ""
echo "[5/7] Exporting model to HuggingFace format..."
python3 export_model.py --checkpoint ./output/checkpoints --output ./output/merged --mode merge

echo ""
echo "[6/7] Setting up llama.cpp for GGUF conversion..."
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
echo "[7/7] Converting to GGUF and quantizing..."
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
echo ""
echo "Run locally with Ollama:"
echo "  ollama create asuka -f Modelfile"
echo "  ollama run asuka"
