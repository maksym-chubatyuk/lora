#!/bin/bash
# =============================================================================
# Qwen 8B LoRA Training Pipeline
# =============================================================================
# Complete automated pipeline: venv -> train -> GGUF -> GCS -> shutdown
# Usage: bash run.sh
# =============================================================================

set -e

# Configuration
PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
BUCKET="gs://${PROJECT_ID}-lora-models"
GGUF_NAME="model-bf16.gguf"

echo "============================================================"
echo "  Qwen 8B LoRA Training Pipeline"
echo "============================================================"
echo ""

if [ -z "$PROJECT_ID" ]; then
    echo "Warning: Could not detect GCP project ID"
    echo "Make sure gcloud is configured or set PROJECT_ID manually"
    PROJECT_ID="my-project"
    BUCKET="gs://${PROJECT_ID}-lora-models"
fi

echo "Project: ${PROJECT_ID}"
echo "Bucket: ${BUCKET}"
echo "Output: ${GGUF_NAME}"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Create GCS bucket if it doesn't exist
# -----------------------------------------------------------------------------
echo "[1/7] Checking GCS bucket..."
if gsutil ls "${BUCKET}" &>/dev/null; then
    echo "  Bucket exists: ${BUCKET}"
else
    echo "  Creating bucket: ${BUCKET}"
    gsutil mb -l us-central1 "${BUCKET}" || {
        echo "  Warning: Could not create bucket. Will try to upload anyway."
    }
fi

# -----------------------------------------------------------------------------
# Step 2: Create virtual environment and install dependencies
# -----------------------------------------------------------------------------
if [ ! -d "venv" ]; then
    echo ""
    echo "[2/7] Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate

    echo ""
    echo "  [2.1] Upgrading pip..."
    pip install --upgrade pip

    echo ""
    echo "  [2.2] Installing PyTorch with CUDA 12.1..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121

    echo ""
    echo "  [2.3] Installing transformers..."
    pip install "transformers>=4.45.0"

    echo ""
    echo "  [2.4] Installing datasets..."
    pip install datasets

    echo ""
    echo "  [2.5] Installing peft (LoRA)..."
    pip install peft

    echo ""
    echo "  [2.6] Installing accelerate..."
    pip install accelerate

    echo ""
    echo "  [2.7] Installing sentencepiece..."
    pip install sentencepiece

    echo ""
    echo "  [2.8] Installing pillow (for VL model)..."
    pip install pillow

    echo ""
    echo "  [2.9] Installing huggingface_hub CLI..."
    pip install huggingface_hub

    echo ""
    echo "  All dependencies installed successfully!"
else
    echo ""
    echo "[2/7] Activating existing virtual environment..."
    source venv/bin/activate
    echo "  Virtual environment activated."
fi

# -----------------------------------------------------------------------------
# Step 2b: HuggingFace Authentication (for downloading models)
# -----------------------------------------------------------------------------
echo ""
echo "[2b/7] Checking HuggingFace authentication..."
if ! huggingface-cli whoami &>/dev/null; then
    echo "  Not logged in. Running huggingface-cli login..."
    echo "  Create a token at: https://huggingface.co/settings/tokens"
    echo ""
    huggingface-cli login
else
    HF_USER=$(huggingface-cli whoami 2>/dev/null | head -1)
    echo "  Logged in as: ${HF_USER}"
fi

# -----------------------------------------------------------------------------
# Step 3: Run training
# -----------------------------------------------------------------------------
echo ""
echo "[3/7] Starting training..."
echo "------------------------------------------------------------"
python3 train.py
echo "------------------------------------------------------------"
echo "  Training complete."

# -----------------------------------------------------------------------------
# Step 4: Setup llama.cpp for GGUF conversion
# -----------------------------------------------------------------------------
echo ""
echo "[4/7] Setting up llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    echo ""
    echo "  [4.1] Cloning llama.cpp repository..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp

    echo ""
    echo "  [4.2] Installing gguf Python package..."
    pip install gguf

    echo ""
    echo "  [4.3] Installing numpy..."
    pip install numpy

    echo ""
    echo "  llama.cpp setup complete!"
else
    echo "  llama.cpp already exists, skipping clone."
fi

# -----------------------------------------------------------------------------
# Step 5: Merge adapters and convert to GGUF
# -----------------------------------------------------------------------------
echo ""
echo "[5/7] Merging adapters and converting to GGUF..."
echo "------------------------------------------------------------"
python3 merge_and_convert.py
echo "------------------------------------------------------------"

# -----------------------------------------------------------------------------
# Step 6: Upload to GCS
# -----------------------------------------------------------------------------
echo ""
echo "[6/7] Uploading to GCS..."
if [ -f "output/${GGUF_NAME}" ]; then
    gsutil cp "output/${GGUF_NAME}" "${BUCKET}/${GGUF_NAME}"
    echo "  Uploaded: ${BUCKET}/${GGUF_NAME}"
else
    echo "  Error: output/${GGUF_NAME} not found!"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 7: Shutdown
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Pipeline Complete!"
echo "============================================================"
echo ""
echo "Your model has been uploaded to:"
echo "  ${BUCKET}/${GGUF_NAME}"
echo ""
echo "Download it locally with:"
echo "  gsutil cp ${BUCKET}/${GGUF_NAME} ."
echo ""
echo "Quantize locally with:"
echo "  ./llama.cpp/build/bin/llama-quantize ${GGUF_NAME} model-q4_k_m.gguf Q4_K_M"
echo ""
echo "Done! Remember to shut down your VM when finished."
