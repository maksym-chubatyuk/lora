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
GGUF_NAME="model-f16.gguf"

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

    echo "  Installing dependencies..."
    pip install --upgrade pip -q
    pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
    pip install "transformers>=4.45.0" datasets peft accelerate sentencepiece -q
    echo "  Dependencies installed."
else
    echo ""
    echo "[2/7] Activating existing virtual environment..."
    source venv/bin/activate
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
    echo "  Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp
    pip install gguf numpy -q
else
    echo "  llama.cpp already exists."
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
echo "[7/7] Shutting down VM in 10 seconds..."
echo "  (Press Ctrl+C to cancel shutdown)"
sleep 10
sudo shutdown -h now
