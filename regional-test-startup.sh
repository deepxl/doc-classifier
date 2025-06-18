#!/bin/bash
set -e

echo "🚀 Starting regional Vertex AI test from us-central1..."

# Update system
apt-get update -y
apt-get install -y python3-pip git python3-venv

# Setup working directory
cd /tmp
rm -rf doc-classifier 2>/dev/null || true

# Clone repository
echo "📥 Cloning DeepXL doc-classifier..."
git clone https://github.com/deepxl/doc-classifier.git
cd doc-classifier

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Setup environment (you'll need to add your real API key)
echo "⚙️ Setting up environment..."
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env.local
echo "GCP_PROJECT_ID=deepxl-backend" >> .env.local

# Run the regional test
echo "🎯 Running Vertex AI regional speed test..."
echo "===============================================" > /tmp/regional-test-results.txt
echo "🚀 VERTEX AI REGIONAL SPEED TEST" >> /tmp/regional-test-results.txt
echo "FROM: us-central1 VM" >> /tmp/regional-test-results.txt
echo "TO: Vertex AI us-central1" >> /tmp/regional-test-results.txt
echo "TIME: $(date)" >> /tmp/regional-test-results.txt
echo "===============================================" >> /tmp/regional-test-results.txt

python3 scripts/speed-only-test.py >> /tmp/regional-test-results.txt 2>&1

echo "✅ Regional test completed!"
echo "📊 Results saved to /tmp/regional-test-results.txt"
echo "📊 To view results: gcloud compute ssh vertex-regional-test --zone=us-central1-a --command='cat /tmp/regional-test-results.txt'" 