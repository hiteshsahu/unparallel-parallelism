#!/bin/bash

echo "🌀 Starting Unparallel Parallelism setup..."

# Go to repo root (one level up from scripts/)
# ----------------------------
# Repo root
# ----------------------------
REPO_ROOT="$(dirname "$(realpath "$0")")/"
cd "$REPO_ROOT"
echo "ℹ️ Repo root: $REPO_ROOT"

# ----------------------------
# 1️⃣ Update Ubuntu
# ----------------------------
echo "🔄 Updating Ubuntu..."
sudo apt update && sudo apt upgrade -y

# ----------------------------
# 2️⃣ Install basic tools
# ----------------------------
echo "🛠 Installing required tools..."
sudo apt install -y build-essential dkms curl gnupg lsb-release apt-transport-https software-properties-common

# ----------------------------
# 3️⃣ Install gcc if missing
# ----------------------------
if ! command -v gcc &> /dev/null
then
    echo "🔧 Installing gcc..."
    sudo apt install -y gcc
fi

# ----------------------------
# 4️⃣ Install NVIDIA Docker dependencies
# ----------------------------
echo "🐳 Installing Docker..."

DOCKER_KEY="/usr/share/keyrings/docker-archive-keyring.gpg"

if [ ! -f "$DOCKER_KEY" ]; then
    echo "⬇️ Adding Docker GPG key..."
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o "$DOCKER_KEY"
else
    echo "ℹ️ Docker GPG key already exists, skipping..."
fi

# Add Docker repository (idempotent)
echo "deb [arch=$(dpkg --print-architecture) signed-by=$DOCKER_KEY] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# ----------------------------
# 5️⃣ Add user to docker group
# ----------------------------
echo "👤 Adding current user to docker group..."
# sudo groupadd docker || true
# sudo usermod -aG docker $USER
# newgrp docker

# ----------------------------
# 6️⃣ Install CUDA Toolkit (skip driver)
# ----------------------------
echo "⚡ Installing CUDA Toolkit (no driver)..."

CUDA_RUNFILE="cuda_13.0.2_580.95.05_linux.run"
CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers//${CUDA_RUNFILE_BASENAME}"


# Download if missing
if [ ! -f "$CUDA_RUNFILE" ]; then
    echo "⬇️ CUDA installer not found, downloading..."
    wget "$CUDA_RUNFILE_URL" -O "$CUDA_RUNFILE"
else
    echo "ℹ️ CUDA installer already exists at $CUDA_RUNFILE"
fi

# Make it executable
chmod +x "$CUDA_RUNFILE"

# Install toolkit
echo "⚡ Installing CUDA Toolkit (no driver)..."
sudo sh "$CUDA_RUNFILE" --silent --toolkit

# ----------------------------
# 7️⃣ Setup environment variables
# ----------------------------
echo "🌐 Setting up CUDA environment variables..."
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc


# ----------------------------
# 8️⃣ Verify installations
# ----------------------------
echo "✅ Verifying installations..."
echo "- GCC version:"
gcc --version
echo "- NVCC version:"
nvcc --version || echo "⚠️ nvcc not found yet, maybe source ~/.bashrc again"
echo "- Docker version:"
docker --version
echo "- NVIDIA GPU access:"
docker run --gpus all --rm nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04 nvidia-smi

echo "🎉 Setup complete! Your src/ directory is ready for project code and you can build/run the CUDA Docker container."
