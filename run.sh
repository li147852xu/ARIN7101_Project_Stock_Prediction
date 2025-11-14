#!/bin/bash

# 股票短期涨跌预测项目快速启动脚本

echo "=========================================="
echo "Stock Price Movement Prediction"
echo "=========================================="
echo ""

# 检查Python版本
echo "Checking Python version..."
python --version
echo ""

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "Virtual environment created."
    echo ""
fi

# 激活虚拟环境
echo "Activating virtual environment..."
source venv/bin/activate
echo ""

# 安装依赖
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed."
echo ""

# 创建必要的目录
echo "Creating directories..."
mkdir -p data/raw data/processed results/models plots logs
touch data/raw/.gitkeep data/processed/.gitkeep
echo "Directories created."
echo ""

# 运行主程序
echo "=========================================="
echo "Starting main program..."
echo "=========================================="
echo ""

python main.py "$@"

echo ""
echo "=========================================="
echo "Program finished!"
echo "=========================================="

