#!/bin/bash
# Generate synthetic training data on GCP A100
# Usage: bash generate_run.sh

set -e

BUCKET="gs://maksym-adapters"
COUNT=10000

echo "=== Synthetic Data Generation (Hermes-4.3-36B on A100) ==="
echo ""

# Step 1: Virtual environment
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate

    echo "[2/4] Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install "transformers>=4.45.0" accelerate bitsandbytes
    pip install sentencepiece protobuf
else
    echo "[1/4] Virtual environment exists, activating..."
    source venv/bin/activate
    echo "[2/4] Dependencies already installed"
fi

echo ""
echo "[3/4] Generating ${COUNT} conversations..."
python3 generate_data.py --character character.txt --output synthetic_data.jsonl --count ${COUNT}

echo ""
echo "[4/4] Uploading to GCS..."
gsutil cp synthetic_data.jsonl $BUCKET/synthetic_data.jsonl
gsutil cp character.txt $BUCKET/character.txt

echo ""
echo "=== Complete! ==="
echo ""
echo "Generated: synthetic_data.jsonl"
echo "Uploaded to: $BUCKET/synthetic_data.jsonl"
echo ""
echo "Download with:"
echo "  gsutil cp $BUCKET/synthetic_data.jsonl ."
