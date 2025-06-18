#!/bin/bash
set -e

echo "🔧 Setting up Python environment on GCP VM..."

# Update system
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Navigate to project
cd ~/gemini-classifier

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables (these should already be set via service account)
export CLOUDSDK_PYTHON=python3.11

echo "🧪 Running Vertex AI speed test from us-central1..."

# Run focused Vertex AI regional speed test
python3.11 scripts/test-vertex-speed.py

echo "✅ Speed test complete! Results saved to results/vertex-ai-us-central1-speed.json"
