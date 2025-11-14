#!/bin/bash
# 完整Demo演示脚本 - 适用于RTX 5090云平台

set -e

echo "=================================================="
echo "Stock Price Prediction - Complete Demo"
echo "Platform: Cloud GPU (RTX 5090)"
echo "=================================================="
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# 激活环境
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Please run deploy_cloud.sh first."
    exit 1
fi

# 显示系统信息
echo "=== System Information ==="
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Python: $(python --version)"
echo "Pip packages:"
pip list | grep -E "torch|numpy|pandas"
echo ""

# 测试GPU
echo "=== GPU Information ==="
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    print(f'CUDA Version: {torch.version.cuda}')
"
echo ""

# 询问Demo类型
echo "Please select demo type:"
echo "  1) Quick Test (5 minutes) - Single stock, 2 models"
echo "  2) Standard Demo (20 minutes) - 5 stocks, 3 models"
echo "  3) Complete Demo (60-90 minutes) - 5 stocks, all models"
echo ""
read -p "Enter your choice [1-3]: " demo_choice

case $demo_choice in
    1)
        echo ""
        echo "Running Quick Test Demo..."
        echo "This will train MLP and LSTM on 1 stock with 10 epochs."
        echo ""
        
        # 创建快速测试配置
        cat > config_quick.yaml << 'EOF'
data:
  stock_codes:
    - "600519.SS"
  start_date: "2023-01-01"
  end_date: null
  data_source: "yfinance"
  raw_data_dir: "data/raw"
  processed_data_dir: "data/processed"

features:
  sequence_length: 20
  prediction_horizon: 1
  indicators:
    ma_periods: [5, 10, 20]
    macd_fast: 12
    macd_slow: 26
    macd_signal: 9
    rsi_period: 14
    bb_period: 20
    bb_std: 2
    atr_period: 14
    stoch_period: 14
    cci_period: 20
    williams_period: 14
    roc_period: 10
    mfi_period: 14

dataset:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

models:
  mlp:
    hidden_dims: [64, 32]
    dropout: 0.3
    activation: "relu"
  
  lstm:
    hidden_dim: 64
    num_layers: 2
    dropout: 0.3
    bidirectional: true
  
  transformer:
    d_model: 64
    nhead: 4
    num_layers: 2
    dim_feedforward: 256
    dropout: 0.1
  
  mamba:
    d_model: 64
    n_layers: 2
    d_state: 16
    expand: 2
    dropout: 0.1

training:
  batch_size: 128
  epochs: 10
  learning_rate: 0.001
  optimizer: "adam"
  scheduler:
    type: "reduce_on_plateau"
    patience: 5
    factor: 0.5
  early_stopping:
    patience: 10
    min_delta: 0.001
  loss: "cross_entropy"
  use_class_weights: true
  seed: 42
  device: "cuda"

evaluation:
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1"
    - "auc"
  save_confusion_matrix: true
  save_roc_curve: true
  results_dir: "results"

visualization:
  enable: true
  plots_dir: "plots"
  format: "png"
  dpi: 150

logging:
  level: "INFO"
  log_dir: "logs"
  save_to_file: true
EOF
        
        python main.py --config config_quick.yaml --step download
        python main.py --config config_quick.yaml --step train --model mlp,lstm
        python main.py --config config_quick.yaml --step evaluate
        ;;
        
    2)
        echo ""
        echo "Running Standard Demo..."
        echo "This will train MLP, LSTM, and Transformer on 5 stocks."
        echo ""
        
        python main.py --step download
        python main.py --step train --model mlp,lstm,transformer
        python main.py --step evaluate
        ;;
        
    3)
        echo ""
        echo "Running Complete Demo..."
        echo "This will train all 6 models on 5 stocks."
        echo "Estimated time: 60-90 minutes"
        echo ""
        read -p "Continue? [y/N]: " confirm
        
        if [[ $confirm != [yY] ]]; then
            echo "Demo cancelled."
            exit 0
        fi
        
        echo ""
        echo "Step 1/5: Downloading data..."
        python main.py --step download
        
        echo ""
        echo "Step 2/5: Training MLP (baseline)..."
        python main.py --step train --model mlp
        
        echo ""
        echo "Step 3/5: Training LSTM..."
        python main.py --step train --model lstm
        
        echo ""
        echo "Step 4/5: Training Transformer..."
        python main.py --step train --model transformer
        
        echo ""
        echo "Step 5/5: Training Mamba..."
        python main.py --step train --model mamba
        
        echo ""
        echo "Evaluating all models..."
        python main.py --step evaluate
        
        echo ""
        echo "Optional: Training statistical models (ARIMA, Prophet)..."
        read -p "Train statistical models? (slow) [y/N]: " train_stat
        
        if [[ $train_stat == [yY] ]]; then
            python main.py --step train --model arima,prophet
        fi
        ;;
        
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# 显示结果
echo ""
echo "=================================================="
echo "Demo Completed Successfully!"
echo "=================================================="
echo ""

echo "Generated files:"
echo "  Models:  $(ls -1 results/models/*.pth 2>/dev/null | wc -l) model files"
echo "  Plots:   $(ls -1 plots/*.png 2>/dev/null | wc -l) visualization files"
echo "  Results: $(ls -1 results/*.csv results/*.txt 2>/dev/null | wc -l) report files"
echo ""

echo "Key Results:"
if [ -f "results/model_comparison.csv" ]; then
    echo ""
    cat results/model_comparison.csv
else
    echo "Model comparison not available yet."
fi

echo ""
echo "Check results in:"
echo "  - results/       (models and reports)"
echo "  - plots/         (visualizations)"
echo "  - logs/          (training logs)"
echo ""

# 尝试显示图表（如果支持图形界面）
if command -v xdg-open &> /dev/null && [ -n "$DISPLAY" ]; then
    echo "Opening result plots..."
    xdg-open plots/model_comparison.png 2>/dev/null || true
fi

echo "=================================================="
echo "Thank you for using Stock Prediction System!"
echo "=================================================="

