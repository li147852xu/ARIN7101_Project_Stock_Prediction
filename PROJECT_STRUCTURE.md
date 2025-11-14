# 项目结构说明

## 目录结构

```
Project/
├── config.yaml                    # 项目配置文件
├── main.py                        # 主执行脚本
├── requirements.txt               # Python依赖
├── README.md                      # 项目说明文档
├── QUICKSTART.md                  # 快速开始指南
├── PROJECT_STRUCTURE.md           # 本文件
├── .gitignore                     # Git忽略文件
├── run.sh                         # Linux/Mac启动脚本
├── run.bat                        # Windows启动脚本
│
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   │   └── .gitkeep
│   └── processed/                 # 处理后的数据
│       └── .gitkeep
│
├── src/                           # 源代码目录
│   ├── __init__.py
│   ├── data_loader.py             # 数据加载模块
│   ├── feature_engineering.py     # 特征工程模块
│   ├── train.py                   # 训练脚本
│   ├── evaluate.py                # 评估脚本
│   │
│   └── models/                    # 模型目录
│       ├── __init__.py
│       ├── mlp.py                 # MLP模型
│       ├── lstm.py                # LSTM模型
│       ├── transformer.py         # Transformer模型
│       ├── mamba_model.py         # Mamba模型
│       ├── arima_model.py         # ARIMA模型
│       └── prophet_model.py       # Prophet模型
│
├── results/                       # 结果目录
│   ├── models/                    # 保存的模型文件
│   ├── *.csv                      # 评估结果CSV
│   └── *.txt                      # 分类报告
│
├── plots/                         # 可视化图表
│   ├── *_confusion_matrix.png     # 混淆矩阵
│   ├── *_roc_curve.png            # ROC曲线
│   ├── *_training_history.png     # 训练历史
│   └── model_comparison.png       # 模型对比
│
└── logs/                          # 日志文件
    └── *.log
```

## 核心文件说明

### 1. 配置和启动

#### `config.yaml`
项目的核心配置文件，包含：
- 数据配置（股票代码、时间范围、数据源）
- 特征工程参数（技术指标参数、序列长度）
- 模型配置（各模型的超参数）
- 训练配置（学习率、batch size、epochs等）
- 评估和可视化配置

#### `main.py`
主执行脚本，整合所有功能：
- 数据加载和预处理
- 模型训练
- 模型评估
- 结果保存

支持命令行参数：
- `--step`: 选择执行步骤（all/download/train/evaluate）
- `--model`: 选择要训练的模型
- `--config`: 指定配置文件路径

### 2. 数据处理

#### `src/data_loader.py`
数据加载模块，负责：
- 从多个数据源下载股票数据（yfinance、akshare）
- 数据清洗和验证
- 本地缓存管理

**主要类**:
- `StockDataLoader`: 数据加载器

**主要函数**:
- `download_stock_data()`: 下载数据
- `clean_data()`: 清洗数据
- `load_data()`: 便捷加载函数

#### `src/feature_engineering.py`
特征工程模块，负责：
- 计算20+种技术指标
- 创建预测目标
- 构造时间序列样本
- 数据切分和标准化
- **防止时间泄漏**

**主要类**:
- `FeatureEngineer`: 特征工程器

**技术指标**:
- 趋势指标: SMA, EMA, MACD, ADX
- 动量指标: RSI, Stochastic, CCI, Williams %R, ROC
- 波动率指标: Bollinger Bands, ATR, Volatility
- 成交量指标: OBV, Volume MA, MFI
- 其他指标: 价格位置、距离高低点等

**关键函数**:
- `calculate_technical_indicators()`: 计算所有技术指标
- `create_target()`: 创建预测目标（防泄漏）
- `create_sequences()`: 构造时间序列样本
- `split_data()`: 按时间顺序切分数据
- `normalize_features()`: 标准化特征

### 3. 模型定义

#### `src/models/mlp.py`
多层感知机（MLP）模型：
- `MLPModel`: 基础MLP
- `MLPModelWithAttention`: 带注意力的MLP
- `create_mlp_model()`: 模型创建函数

#### `src/models/lstm.py`
长短期记忆网络（LSTM）模型：
- `LSTMModel`: 标准LSTM
- `LSTMAttentionModel`: 带注意力的LSTM
- `StackedLSTMModel`: 堆叠LSTM（带残差连接）
- `create_lstm_model()`: 模型创建函数

#### `src/models/transformer.py`
Transformer模型：
- `PositionalEncoding`: 位置编码
- `TransformerModel`: 标准Transformer编码器
- `TransformerWithCLS`: 带CLS token的Transformer
- `TimeSeriesTransformer`: 时序专用Transformer
- `create_transformer_model()`: 模型创建函数

#### `src/models/mamba_model.py`
Mamba状态空间模型：
- `MambaModel`: 标准Mamba模型
- `SimpleMambaModel`: 简化版（使用GRU作为替代）
- `HybridMambaModel`: 混合模型（Mamba + Attention）
- `create_mamba_model()`: 模型创建函数

#### `src/models/arima_model.py`
ARIMA统计模型：
- `ARIMAModel`: 单变量ARIMA
- `ARIMAEnsemble`: 多股票集成ARIMA
- `create_arima_model()`: 模型创建函数

#### `src/models/prophet_model.py`
Prophet时序模型：
- `ProphetModel`: 基础Prophet
- `ProphetEnsemble`: 多股票集成Prophet
- `create_prophet_model()`: 模型创建函数

### 4. 训练和评估

#### `src/train.py`
模型训练模块：
- `Trainer`: 训练器类
  - 支持多种优化器（Adam, AdamW, SGD）
  - 支持多种损失函数（Cross Entropy, Focal Loss）
  - 学习率调度（ReduceLROnPlateau, CosineAnnealing）
  - 早停机制
  - 梯度裁剪
  - 模型保存和加载

- `FocalLoss`: Focal Loss损失函数（处理类别不平衡）
- `train_model()`: 便捷训练函数

**训练流程**:
1. 创建数据加载器
2. 前向传播
3. 计算损失
4. 反向传播和梯度更新
5. 验证和早停检查
6. 保存最佳模型

#### `src/evaluate.py`
模型评估模块：
- `Evaluator`: 评估器类
  - 计算多种评估指标
  - 生成混淆矩阵
  - 绘制ROC曲线
  - 保存分类报告
  - 模型对比

**评估指标**:
- Accuracy（准确率）
- Precision（精确率）
- Recall（召回率）
- F1-Score（F1分数）
- AUC（ROC曲线下面积）
- Confusion Matrix（混淆矩阵）

**可视化**:
- 混淆矩阵热力图
- ROC曲线
- 训练历史曲线
- 模型对比柱状图

### 5. 辅助文件

#### `requirements.txt`
Python依赖包列表：
- 数据处理: pandas, numpy
- 数据获取: yfinance, akshare
- 技术指标: ta, talib-binary
- 机器学习: scikit-learn, torch, pytorch-lightning
- 时序模型: statsmodels, prophet
- Mamba: mamba-ssm, causal-conv1d
- 可视化: matplotlib, seaborn, plotly

#### `run.sh` / `run.bat`
自动化启动脚本：
- 检查Python版本
- 创建虚拟环境
- 安装依赖
- 创建目录
- 运行主程序

## 数据流程

```
1. 数据下载
   └─> StockDataLoader.download_stock_data()
       └─> 从yfinance/akshare获取OHLCV数据
       └─> 数据清洗和验证
       └─> 保存到data/raw/

2. 特征工程
   └─> FeatureEngineer.calculate_technical_indicators()
       └─> 计算20+种技术指标
       └─> 创建目标变量（防泄漏）
       └─> 构造时间序列样本
       └─> 按时间切分train/val/test
       └─> 标准化特征
       └─> 保存到data/processed/

3. 模型训练
   └─> 对每个模型：
       └─> 创建模型实例
       └─> Trainer.train()
           └─> 前向传播
           └─> 计算损失
           └─> 反向传播
           └─> 验证和早停
       └─> 保存到results/models/

4. 模型评估
   └─> 对每个模型：
       └─> Evaluator.evaluate_model()
           └─> 预测测试集
           └─> 计算评估指标
           └─> 生成可视化
           └─> 保存报告
       └─> Evaluator.compare_models()
           └─> 对比所有模型
           └─> 生成对比图表
```

## 防止时间泄漏的措施

1. **严格时间顺序**:
   - 数据切分严格按时间顺序
   - 训练集 → 验证集 → 测试集
   - 不打乱数据

2. **特征计算**:
   - 所有技术指标只使用历史数据
   - 使用`.shift()`确保不使用当日或未来数据
   - 移动平均等指标正确计算

3. **标准化**:
   - 只在训练集上计算均值和标准差
   - 用训练集的统计量转换验证集和测试集

4. **目标构造**:
   - 使用`.shift(-1)`获取下一日数据
   - 确保预测目标是未来信息

## 扩展指南

### 添加新的技术指标

在 `src/feature_engineering.py` 中：

```python
def _calculate_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    # 添加你的指标
    df['custom_indicator'] = ...
    return df
```

### 添加新模型

1. 在 `src/models/` 下创建新文件 `your_model.py`
2. 实现模型类和 `create_your_model()` 函数
3. 在 `main.py` 中导入和使用

### 修改训练流程

在 `src/train.py` 中修改 `Trainer` 类的方法。

### 自定义评估指标

在 `src/evaluate.py` 中的 `Evaluator._calculate_metrics()` 添加新指标。

## 最佳实践

1. **开始前**:
   - 阅读 QUICKSTART.md
   - 检查 config.yaml 配置
   - 确保网络连接稳定

2. **实验管理**:
   - 修改配置前备份
   - 记录实验参数和结果
   - 使用Git版本控制

3. **性能优化**:
   - 使用GPU（如果可用）
   - 调整batch_size
   - 使用早停避免过拟合

4. **调试**:
   - 检查logs/目录下的日志
   - 使用小数据集快速测试
   - 逐步增加复杂度

## 常用命令速查

```bash
# 完整流程
python main.py

# 只下载数据
python main.py --step download

# 训练特定模型
python main.py --step train --model mlp,lstm

# 只评估
python main.py --step evaluate

# 使用自定义配置
python main.py --config my_config.yaml
```

## 注意事项

1. 首次运行会下载数据，需要时间
2. Mamba模型需要CUDA，否则使用GRU替代
3. 数据量大时需要较多内存
4. 建议使用GPU进行训练
5. 注意数据版权和使用限制

## 技术栈

- **语言**: Python 3.8+
- **深度学习**: PyTorch
- **数据处理**: pandas, numpy
- **技术指标**: TA-Lib, ta
- **时序模型**: statsmodels, prophet
- **可视化**: matplotlib, seaborn
- **数据源**: yfinance, akshare

## 参考文献

- LSTM: Hochreiter & Schmidhuber (1997)
- Transformer: Vaswani et al. (2017)
- Mamba: Gu & Dao (2023)
- ARIMA: Box & Jenkins (1970)
- Prophet: Taylor & Letham (2018)

