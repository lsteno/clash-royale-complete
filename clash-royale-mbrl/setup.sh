#!/bin/bash
# Setup script for Clash Royale Visual-MBRL on Apple Silicon

set -e

echo "=================================="
echo "Clash Royale Visual-MBRL Setup"
echo "Apple Silicon Edition"
echo "=================================="

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "Warning: This setup is optimized for Apple Silicon (ARM64)"
    echo "Current architecture: $(uname -m)"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$PYTHON_VERSION" < "3.10" ]]; then
    echo "Error: Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION detected"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with MPS support
echo ""
echo "Installing PyTorch (MPS enabled)..."
pip install torch torchvision torchaudio

# Verify MPS
echo ""
echo "Verifying MPS support..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"

# Install other requirements
echo ""
echo "Installing project dependencies..."
pip install -r requirements-apple-silicon.txt

# Check for Homebrew
echo ""
echo "Checking system dependencies..."

if ! command -v brew &> /dev/null; then
    echo "Warning: Homebrew not found. Install from https://brew.sh"
else
    # Check/install ADB
    if ! command -v adb &> /dev/null; then
        echo "Installing Android platform tools (ADB)..."
        brew install android-platform-tools
    else
        echo "✓ ADB installed"
    fi
    
    # Check/install scrcpy
    if ! command -v scrcpy &> /dev/null; then
        echo "Installing scrcpy..."
        brew install scrcpy
    else
        echo "✓ scrcpy installed"
    fi
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p checkpoints
mkdir -p weights
mkdir -p logs
mkdir -p data/replays
echo "✓ Directories created"

# Download YOLOv8 base model
echo ""
echo "Downloading YOLOv8 base model..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
echo "✓ YOLOv8n downloaded"

# Test imports
echo ""
echo "Testing project imports..."
python -c "
from src.utils.device import get_device, check_mps_capabilities
from src.agent import DreamerV3Model, DreamerConfig
from src.perception import PerceptionPipeline

print('✓ All imports successful')
print(f'✓ Default device: {get_device()}')
"

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Install Android Studio and create an ARM64 emulator"
echo "2. Install Clash Royale on the emulator"
echo "3. (Optional) Download KataCR YOLOv8 weights from:"
echo "   https://drive.google.com/drive/folders/..."
echo "4. Run test training:"
echo "   python train.py --no-emulator --total-steps 1000"
echo ""
