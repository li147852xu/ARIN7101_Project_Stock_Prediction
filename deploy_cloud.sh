#!/bin/bash
# RTX 5090云平台快速部署脚本

set -e

echo "=================================================="
echo "Stock Prediction System - Cloud GPU Deployment"
echo "GPU: NVIDIA RTX 5090"
echo "=================================================="
echo ""

# 1. 克隆项目
echo "Step 1: Cloning repository..."
if [ ! -d "ARIN7101_Project_Stock_Prediction" ]; then
    git clone https://github.com/li147852xu/ARIN7101_Project_Stock_Prediction.git
fi
cd ARIN7101_Project_Stock_Prediction

# 2. 创建虚拟环境
echo ""
echo "Step 2: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# 3. 升级pip
echo ""
echo "Step 3: Upgrading pip..."
pip install --upgrade pip

# 4. 安装PyTorch (CUDA 12.1)
echo ""
echo "Step 4: Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. 安装其他依赖
echo ""
echo "Step 5: Installing other dependencies..."
pip install -r requirements.txt

# 6. 安装可选依赖
echo ""
echo "Step 6: Installing optional dependencies..."
pip install yfinance akshare ta prophet statsmodels

# 7. 尝试安装Mamba（可选）
echo ""
echo "Step 7: Installing Mamba (optional, may fail)..."
pip install mamba-ssm causal-conv1d || echo "Mamba installation skipped, will use GRU fallback"

# 8. 验证安装
echo ""
echo "Step 8: Verifying installation..."
python test_setup.py

# 9. 测试GPU
echo ""
echo "Step 9: Testing GPU availability..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB' if torch.cuda.is_available() else 'N/A')"

echo ""
echo "=================================================="
echo "Deployment completed successfully!"
echo "=================================================="
echo ""
echo "Quick Start:"
echo "  cd ARIN7101_Project_Stock_Prediction"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "Or run demo:"
echo "  ./demo_complete.sh"
echo ""

