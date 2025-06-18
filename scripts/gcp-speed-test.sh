#!/bin/bash
# GCP Speed Test - Run from us-central1 to get realistic Vertex AI performance

set -e

echo "🚀 Setting up GCP Compute Engine instance for Vertex AI speed testing..."

# Configuration - Get current project automatically
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No active GCP project found. Please run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

INSTANCE_NAME="gemini-speed-test"
ZONE="us-central1-a"
MACHINE_TYPE="e2-standard-2"  # 2 vCPUs, 8GB RAM - good for parallel testing

echo "📋 Using project: $PROJECT_ID"

# Create VM instance in us-central1
echo "📦 Creating VM instance in us-central1..."
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=admin-geir@deepxl-backend.iam.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --create-disk=auto-delete=yes,boot=yes,device-name=$INSTANCE_NAME,image-family=ubuntu-2204-lts,image-project=ubuntu-os-cloud,mode=rw,size=20,type=projects/$PROJECT_ID/zones/$ZONE/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=purpose=speed-testing \
    --reservation-affinity=any

echo "⏳ Waiting for VM to be ready..."
sleep 60

# Wait for SSH to be available
echo "🔗 Waiting for SSH to be ready..."
for i in {1..12}; do
    if gcloud compute ssh --zone=$ZONE $INSTANCE_NAME --command="echo 'SSH ready'" >/dev/null 2>&1; then
        echo "✅ SSH is ready!"
        break
    fi
    echo "   Attempt $i/12: SSH not ready, waiting 10s..."
    sleep 10
done

# Copy project files to VM
echo "📁 Copying project files to VM..."
# Create directory first, then copy essential files only
gcloud compute ssh --zone=$ZONE $INSTANCE_NAME --command="mkdir -p ~/gemini-classifier"
# Copy only necessary directories and files
gcloud compute scp --recurse --zone=$ZONE src/ $INSTANCE_NAME:~/gemini-classifier/
gcloud compute scp --recurse --zone=$ZONE scripts/ $INSTANCE_NAME:~/gemini-classifier/
gcloud compute scp --recurse --zone=$ZONE test-images/ $INSTANCE_NAME:~/gemini-classifier/
gcloud compute scp --zone=$ZONE requirements.txt $INSTANCE_NAME:~/gemini-classifier/
gcloud compute scp --zone=$ZONE .env.local $INSTANCE_NAME:~/gemini-classifier/ 2>/dev/null || echo "   ⚠️  .env.local not found, using service account auth"

# Setup script to run on VM
cat > setup_and_test.sh << 'EOF'
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
EOF

# Copy and run setup script
echo "🔧 Running setup and test on VM..."
gcloud compute scp --zone=$ZONE setup_and_test.sh $INSTANCE_NAME:~/
gcloud compute ssh --zone=$ZONE $INSTANCE_NAME --command="chmod +x ~/setup_and_test.sh && ~/setup_and_test.sh"

# Copy results back
echo "📊 Copying results back..."
gcloud compute scp --zone=$ZONE $INSTANCE_NAME:~/gemini-classifier/results/vertex-ai-us-central1-speed.json ./results/gcp-us-central1-speed-test.json

echo "🎯 GCP us-central1 speed test complete!"
echo "📁 Results saved to: ./results/gcp-us-central1-speed-test.json"

# Optional: Keep VM for more testing or delete it
read -p "🗑️  Delete the test VM? (y/n): " -n 1 -r
echo
if [[ $REPL =~ ^[Yy]$ ]]; then
    echo "🗑️  Deleting VM instance..."
    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
    echo "✅ VM deleted"
else
    echo "💡 VM kept for additional testing. Delete manually when done:"
    echo "   gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
fi 