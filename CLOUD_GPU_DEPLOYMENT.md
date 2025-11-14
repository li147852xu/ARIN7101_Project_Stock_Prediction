# 云平台RTX 5090运行完整Demo方案

## 📋 目录

1. [环境准备](#环境准备)
2. [快速部署](#快速部署)
3. [完整Demo运行](#完整demo运行)
4. [性能优化建议](#性能优化建议)
5. [常见问题](#常见问题)

---

## 🚀 环境准备

### 1. 云平台规格推荐

**推荐配置**（RTX 5090）：
```
GPU:     NVIDIA RTX 5090 (24GB VRAM)
CPU:     16核心以上
内存:    32GB+
存储:    100GB+ SSD
系统:    Ubuntu 20.04/22.04 LTS
CUDA:    12.1+
```

**支持的云平台**：
- AutoDL (推荐，性价比高)
- 恒源云
- 矩池云
- AWS EC2 (g5/p4 实例)
- Google Cloud (A100/V100)
- 阿里云PAI-DSW

### 2. 系统环境检查

登录云平台后，首先检查GPU状态：

```bash
# 检查GPU
nvidia-smi

# 检查CUDA版本
nvcc --version

# 检查系统信息
uname -a
cat /etc/os-release
```

---

## 🎯 快速部署

### 方案A：一键部署脚本（推荐）

创建部署脚本 `deploy_cloud.sh`：

```bash
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
git clone https://github.com/li147852xu/ARIN7101_Project_Stock_Prediction.git
cd ARIN7101_Project_Stock_Prediction

# 2. 创建虚拟环境
echo ""
echo "Step 2: Creating virtual environment..."
python3 -m venv venv
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
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

echo ""
echo "=================================================="
echo "Deployment completed successfully!"
echo "Run demo with: python main.py"
echo "=================================================="
```

运行部署：

```bash
# 下载并运行部署脚本
wget https://raw.githubusercontent.com/li147852xu/ARIN7101_Project_Stock_Prediction/main/deploy_cloud.sh
chmod +x deploy_cloud.sh
./deploy_cloud.sh
```

### 方案B：手动部署（更可控）

```bash
# 1. 克隆项目
git clone https://github.com/li147852xu/ARIN7101_Project_Stock_Prediction.git
cd ARIN7101_Project_Stock_Prediction

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装PyTorch (根据CUDA版本选择)
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 测试配置
python test_setup.py
```

---

## 🎬 完整Demo运行

### Demo 1: 快速测试（5分钟）

**适用场景**：验证环境，快速测试

```bash
# 修改配置以加速
cat > config_quick.yaml << 'EOF'
data:
  stock_codes:
    - "600519.SS"  # 仅测试1只股票
  start_date: "2023-01-01"
  end_date: null
  data_source: "yfinance"
  raw_data_dir: "data/raw"
  processed_data_dir: "data/processed"

features:
  sequence_length: 20  # 减少序列长度
  prediction_horizon: 1
  indicators:
    ma_periods: [5, 10, 20]  # 减少指标
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
    hidden_dims: [64, 32]  # 减小模型
    dropout: 0.3
    activation: "relu"
  
  lstm:
    hidden_dim: 64
    num_layers: 2
    dropout: 0.3
    bidirectional: true

training:
  batch_size: 128  # 增大batch
  epochs: 10       # 减少epochs
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

# 运行快速测试
python main.py --config config_quick.yaml --step train --model mlp,lstm
```

**预期时间**：3-5分钟

### Demo 2: 标准Demo（20分钟）

**适用场景**：完整展示，中等规模

```bash
# 使用默认配置，选择部分模型
python main.py --step download  # 下载数据 (~2分钟)
python main.py --step train --model mlp,lstm,transformer  # 训练 (~15分钟)
python main.py --step evaluate  # 评估 (~3分钟)
```

**预期结果**：
- 训练3个深度学习模型
- 生成完整的评估报告
- 可视化图表

### Demo 3: 完整Demo（60-90分钟）

**适用场景**：完整实验，所有模型对比

#### 3.1 修改配置以充分利用GPU

编辑 `config.yaml`：

```yaml
training:
  batch_size: 256      # RTX 5090可以使用更大batch
  epochs: 50           # 充分训练
  device: "cuda"

data:
  stock_codes:
    - "600519.SS"
    - "600036.SS"
    - "601318.SS"
    - "600030.SS"
    - "600887.SS"
```

#### 3.2 运行完整流程

```bash
# 方式1: 一次性运行所有
python main.py

# 方式2: 分步运行（推荐，便于监控）
# Step 1: 下载和处理数据
python main.py --step download

# Step 2: 训练所有深度学习模型
python main.py --step train --model mlp,lstm,transformer,mamba

# Step 3: 评估所有模型
python main.py --step evaluate

# Step 4: 如果需要，训练统计模型
# 注意：ARIMA和Prophet较慢，可以单独运行
python main.py --step train --model arima
python main.py --step train --model prophet
```

#### 3.3 监控GPU使用

在另一个终端窗口：

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或使用更详细的监控
nvidia-smi dmon -s pucvmet
```

#### 3.4 预期输出

```
results/
├── models/                          # 训练好的模型
│   ├── mlp.pth
│   ├── lstm.pth
│   ├── transformer.pth
│   └── mamba.pth
├── model_comparison.csv             # 模型对比表
└── *_classification_report.txt      # 各模型详细报告

plots/
├── model_comparison.png             # 模型对比图
├── mlp_confusion_matrix.png         # 混淆矩阵
├── mlp_roc_curve.png               # ROC曲线
├── mlp_training_history.png        # 训练历史
└── ... (其他模型的图表)

logs/
└── *.log                            # 运行日志
```

---

## 💡 性能优化建议

### 1. GPU内存优化

```python
# 如果遇到OOM (Out of Memory)，修改config.yaml

training:
  batch_size: 128  # 从256降低到128
  
models:
  lstm:
    hidden_dim: 96   # 从128降低
  
  transformer:
    d_model: 96      # 从128降低
    num_layers: 2    # 从3降低
```

### 2. 数据加载优化

```yaml
# 启用数据缓存
data:
  use_cache: true  # 第二次运行时会更快
  
features:
  sequence_length: 30  # 不要设置太长
```

### 3. 混合精度训练（加速2倍）

创建 `config_fp16.yaml`：

```yaml
training:
  use_amp: true      # 启用自动混合精度
  batch_size: 512    # 可以使用更大的batch
```

然后在 `src/train.py` 中添加AMP支持（已预留接口）。

### 4. 多GPU训练（如有多卡）

```bash
# 使用DataParallel
export CUDA_VISIBLE_DEVICES=0,1
python main.py --config config_multigpu.yaml
```

### 5. 并行数据预处理

```yaml
# config.yaml
training:
  num_workers: 4  # 数据加载线程数
```

---

## 📊 性能基准测试

### RTX 5090预期性能

| 模型 | Batch Size | 训练时间/Epoch | 推理速度 | 显存占用 |
|------|-----------|---------------|---------|---------|
| MLP | 256 | ~10秒 | 5000 samples/s | ~2GB |
| LSTM | 256 | ~30秒 | 2000 samples/s | ~4GB |
| Transformer | 256 | ~45秒 | 1500 samples/s | ~6GB |
| Mamba | 256 | ~35秒 | 2500 samples/s | ~5GB |

### 完整Demo时间估算

| 步骤 | 时间 | 说明 |
|-----|------|-----|
| 数据下载 | 2-5分钟 | 取决于网络速度 |
| 特征计算 | 1-2分钟 | 40+个指标 |
| MLP训练 | 5-10分钟 | 50 epochs |
| LSTM训练 | 15-20分钟 | 50 epochs |
| Transformer训练 | 20-30分钟 | 50 epochs |
| Mamba训练 | 15-25分钟 | 50 epochs |
| 评估 | 2-3分钟 | 所有模型 |
| **总计** | **60-90分钟** | 完整流程 |

---

## 🎓 Demo演示脚本

### 完整演示流程

```bash
#!/bin/bash
# demo_complete.sh - 完整Demo演示脚本

echo "=================================================="
echo "Stock Price Prediction - Complete Demo"
echo "Platform: Cloud GPU (RTX 5090)"
echo "=================================================="
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# 激活环境
source venv/bin/activate

echo "Step 1: Testing GPU..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"

echo ""
echo "Step 2: Downloading data..."
python main.py --step download

echo ""
echo "Step 3: Training MLP (baseline)..."
python main.py --step train --model mlp

echo ""
echo "Step 4: Training LSTM..."
python main.py --step train --model lstm

echo ""
echo "Step 5: Training Transformer..."
python main.py --step train --model transformer

echo ""
echo "Step 6: Training Mamba..."
python main.py --step train --model mamba

echo ""
echo "Step 7: Evaluating all models..."
python main.py --step evaluate

echo ""
echo "=================================================="
echo "Demo completed!"
echo "Results saved to: results/"
echo "Plots saved to: plots/"
echo "=================================================="
echo ""
echo "Key results:"
cat results/model_comparison.csv

# 显示可视化（如果支持X11转发）
if [ -n "$DISPLAY" ]; then
    echo ""
    echo "Opening result plots..."
    xdg-open plots/model_comparison.png
fi
```

运行Demo：

```bash
chmod +x demo_complete.sh
./demo_complete.sh
```

---

## 🔧 实时监控方案

### 方案1: TensorBoard集成

创建 `tensorboard_monitor.py`：

```python
#!/usr/bin/env python
"""
TensorBoard监控脚本
"""
from torch.utils.tensorboard import SummaryWriter
import subprocess
import sys

def start_tensorboard():
    writer = SummaryWriter('runs/stock_prediction')
    print("TensorBoard started at: http://localhost:6006")
    subprocess.Popen(['tensorboard', '--logdir', 'runs', '--bind_all'])
    
if __name__ == '__main__':
    start_tensorboard()
```

运行：

```bash
python tensorboard_monitor.py &
# 访问: http://[云服务器IP]:6006
```

### 方案2: 实时日志查看

```bash
# 在训练的同时，另开一个终端
tail -f logs/*.log

# 或使用更友好的工具
pip install loguru
# 然后查看彩色日志
```

### 方案3: GPU监控脚本

创建 `monitor_gpu.sh`：

```bash
#!/bin/bash
# GPU监控脚本

echo "Monitoring GPU usage..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    date
    echo ""
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
    echo ""
    echo "Training processes:"
    ps aux | grep python | grep main.py
    sleep 2
done
```

---

## ❓ 常见问题

### Q1: CUDA Out of Memory

**解决方案**：

```bash
# 减小batch size
# 编辑 config.yaml:
training:
  batch_size: 64  # 从256降到64
```

### Q2: Mamba安装失败

**解决方案**：

```bash
# 使用预编译版本
pip install mamba-ssm --no-build-isolation

# 或跳过Mamba，系统会自动使用GRU替代
# 无需任何修改
```

### Q3: 数据下载失败

**解决方案**：

```bash
# 方案1: 使用akshare（国内源）
# 修改 config.yaml:
data:
  data_source: "akshare"

# 方案2: 使用代理
export http_proxy=http://proxy_address:port
export https_proxy=http://proxy_address:port
```

### Q4: 训练速度慢

**解决方案**：

```bash
# 1. 确认使用GPU
python -c "import torch; print(torch.cuda.is_available())"

# 2. 增大batch size
# config.yaml:
training:
  batch_size: 256  # RTX 5090可以更大

# 3. 减少数据量（测试用）
data:
  stock_codes: ["600519.SS"]  # 只用1只股票
```

### Q5: 远程访问可视化结果

**解决方案**：

```bash
# 方案1: 使用Jupyter
pip install jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# 方案2: 使用HTTP服务器
cd plots
python -m http.server 8000
# 访问: http://[服务器IP]:8000

# 方案3: 下载到本地
scp -r user@server:/path/to/plots ./local_plots
```

---

## 📦 Docker部署（可选）

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 克隆项目
RUN git clone https://github.com/li147852xu/ARIN7101_Project_Stock_Prediction.git
WORKDIR /workspace/ARIN7101_Project_Stock_Prediction

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir yfinance akshare ta prophet statsmodels

# 配置
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# 入口点
CMD ["python", "main.py"]
```

构建和运行：

```bash
# 构建镜像
docker build -t stock-prediction:latest .

# 运行容器
docker run --gpus all -v $(pwd)/results:/workspace/results stock-prediction:latest
```

---

## 🎯 完整Demo检查清单

运行Demo前检查：

- [ ] GPU可用 (`nvidia-smi`)
- [ ] CUDA版本正确 (`nvcc --version`)
- [ ] Python环境就绪 (`python --version`)
- [ ] 依赖已安装 (`python test_setup.py`)
- [ ] 网络连接正常 (下载数据用)
- [ ] 磁盘空间充足 (至少10GB)

运行Demo后验证：

- [ ] 数据成功下载 (`data/raw/` 有文件)
- [ ] 模型训练完成 (`results/models/` 有.pth文件)
- [ ] 评估报告生成 (`results/` 有CSV和TXT)
- [ ] 可视化图表生成 (`plots/` 有PNG图片)
- [ ] 日志文件正常 (`logs/` 有日志)

---

## 📞 技术支持

如遇问题：

1. **查看日志**：`cat logs/*.log`
2. **检查GPU**：`nvidia-smi`
3. **测试配置**：`python test_setup.py`
4. **查看文档**：项目根目录下的各个.md文件

---

## 🎉 总结

本方案提供了三种Demo运行方式：

1. **快速测试** (5分钟) - 验证环境
2. **标准Demo** (20分钟) - 常规展示
3. **完整Demo** (60-90分钟) - 完整实验

选择适合你时间和需求的方案即可！

---

**文档版本**: v1.0  
**最后更新**: 2024-11-14  
**平台**: RTX 5090 / CUDA 12.1+  
**项目地址**: https://github.com/li147852xu/ARIN7101_Project_Stock_Prediction

