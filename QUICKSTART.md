# 快速开始指南

本指南帮助你快速上手股票短期涨跌预测项目。

## 环境要求

- Python 3.8+
- pip
- 网络连接（用于下载数据）

## 安装步骤

### 方法1：使用脚本（推荐）

#### Linux/Mac:
```bash
chmod +x run.sh
./run.sh
```

#### Windows:
```cmd
run.bat
```

### 方法2：手动安装

1. **创建虚拟环境**
```bash
python -m venv venv
```

2. **激活虚拟环境**

Linux/Mac:
```bash
source venv/bin/activate
```

Windows:
```cmd
venv\Scripts\activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

**注意**: 如果安装mamba-ssm遇到问题（需要CUDA），可以跳过或使用以下命令：
```bash
pip install --no-build-isolation mamba-ssm
```

如果仍然失败，项目会自动使用GRU作为Mamba的替代。

## 运行项目

### 完整流程
```bash
python main.py
```

这将执行：
1. 下载股票数据
2. 计算技术指标
3. 训练所有模型
4. 评估并生成报告

### 分步执行

#### 1. 仅下载数据
```bash
python main.py --step download
```

#### 2. 仅训练模型
```bash
python main.py --step train
```

训练特定模型：
```bash
python main.py --step train --model mlp,lstm
```

#### 3. 仅评估
```bash
python main.py --step evaluate
```

## 配置修改

编辑 `config.yaml` 来修改：

### 股票代码
```yaml
data:
  stock_codes:
    - "600519.SS"  # 贵州茅台
    - "600036.SS"  # 招商银行
    # 添加更多股票...
```

### 时间范围
```yaml
data:
  start_date: "2019-01-01"
  end_date: null  # null表示到最新
```

### 模型参数
```yaml
models:
  mlp:
    hidden_dims: [128, 64, 32]
    dropout: 0.3
  
  lstm:
    hidden_dim: 128
    num_layers: 2
    bidirectional: true
```

### 训练参数
```yaml
training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  device: "cuda"  # 或 "cpu"
```

## 查看结果

训练完成后，结果保存在以下位置：

- **模型文件**: `results/models/`
- **评估报告**: `results/`
- **可视化图表**: `plots/`
- **训练日志**: `logs/`

### 主要结果文件

1. **模型比较** (`results/model_comparison.csv`)
   - 各模型的性能对比

2. **分类报告** (`results/*_classification_report.txt`)
   - 每个模型的详细指标

3. **混淆矩阵** (`plots/*_confusion_matrix.png`)
   - 预测结果的混淆矩阵

4. **ROC曲线** (`plots/*_roc_curve.png`)
   - 模型的ROC曲线和AUC值

5. **训练历史** (`plots/*_training_history.png`)
   - 训练过程的损失和准确率曲线

## 使用Jupyter Notebook

项目提供了示例Notebook：

```bash
jupyter notebook example.ipynb
```

## 常见问题

### 1. 数据下载失败
- 检查网络连接
- 尝试切换数据源（在config.yaml中修改data_source）
- 使用VPN

### 2. CUDA out of memory
- 减小batch_size
- 使用CPU：在config.yaml中设置 `device: "cpu"`
- 减少模型大小

### 3. Mamba安装失败
- 项目会自动使用GRU作为替代
- 或者跳过Mamba模型训练

### 4. 训练太慢
- 减少epochs
- 使用GPU
- 只训练部分模型：`--model mlp,lstm`

## 性能优化建议

1. **使用GPU**: 在config.yaml中设置 `device: "cuda"`
2. **增加batch_size**: 如果GPU内存足够
3. **使用更少的股票**: 减少stock_codes
4. **缩短时间范围**: 修改start_date
5. **减少特征**: 在feature_engineering.py中注释掉部分指标

## 进一步学习

- 阅读 `README.md` 了解项目详情
- 查看 `src/` 目录下的代码
- 修改模型结构和参数进行实验
- 添加自己的技术指标

## 技术支持

如遇问题，请检查：
1. Python版本 >= 3.8
2. 所有依赖已正确安装
3. 数据目录有写入权限
4. 日志文件 (`logs/`) 中的错误信息

## 下一步

1. 尝试不同的股票组合
2. 调整模型超参数
3. 添加新的技术指标
4. 实现自己的模型
5. 进行回测分析

祝你使用愉快！📈

