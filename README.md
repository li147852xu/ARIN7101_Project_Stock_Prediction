# 股票短期涨跌预测项目

## 项目简介

本项目利用历史价格与技术指标数据，预测个股下一交易日的涨跌方向（二分类任务）。

## 主要特点

- **多数据源支持**：支持从yfinance、akshare等数据源获取上证50成分股数据
- **丰富的技术指标**：包含20+种经典技术指标（MA、EMA、MACD、RSI、Bollinger Bands、ATR、OBV等）
- **防止时间泄漏**：严格的时间序列切分，确保训练集不包含未来信息
- **多种模型**：
  - 传统机器学习：MLP（基线）
  - 深度时序模型：LSTM、Transformer、Mamba
  - 统计时序模型：ARIMA、Prophet
- **完整评估**：提供Accuracy、Precision、Recall、F1-Score、AUC等多维度指标

## 项目结构

```
Project/
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   └── processed/             # 处理后的数据
├── src/                       # 源代码
│   ├── data_loader.py        # 数据获取
│   ├── feature_engineering.py # 特征工程
│   ├── models/               # 模型定义
│   │   ├── mlp.py
│   │   ├── lstm.py
│   │   ├── transformer.py
│   │   ├── mamba.py
│   │   ├── arima.py
│   │   └── prophet_model.py
│   ├── train.py              # 训练脚本
│   └── evaluate.py           # 评估脚本
├── config.yaml               # 配置文件
├── main.py                   # 主执行脚本
├── requirements.txt          # 依赖文件
└── README.md                 # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：如果安装mamba-ssm遇到问题，可以尝试：
```bash
pip install mamba-ssm --no-build-isolation
```

### 2. 运行完整流程

```bash
python main.py
```

这将执行以下步骤：
1. 下载数据（贵州茅台、招商银行、中国平安等）
2. 计算技术指标
3. 构造时间序列样本
4. 训练多个模型
5. 评估并生成报告

### 3. 单独执行某个步骤

```bash
# 仅下载数据
python main.py --step download

# 仅训练模型
python main.py --step train --model lstm

# 仅评估
python main.py --step evaluate
```

## 技术指标说明

项目实现了以下技术指标：

### 趋势指标
- SMA/EMA：简单/指数移动平均线（5, 10, 20, 30, 60日）
- MACD：移动平均收敛发散指标
- ADX：平均趋向指数

### 动量指标
- RSI：相对强弱指数
- Stochastic：随机指标
- CCI：商品通道指数
- Williams %R：威廉指标
- ROC：变动速率

### 波动率指标
- Bollinger Bands：布林带
- ATR：平均真实波动幅度
- Standard Deviation：标准差

### 成交量指标
- OBV：能量潮指标
- Volume MA：成交量移动平均
- MFI：资金流量指数

### 其他指标
- 涨跌幅、收益率
- 高低价差、振幅
- 换手率相关指标

## 模型说明

### 1. MLP (多层感知机)
- 简单的前馈神经网络
- 作为基线模型

### 2. LSTM (长短期记忆网络)
- 经典的时序深度学习模型
- 擅长捕捉长期依赖

### 3. Transformer
- 基于注意力机制的模型
- 并行化处理时序数据

### 4. Mamba
- 最新的状态空间模型
- 高效的长序列建模

### 5. ARIMA
- 经典统计时序模型
- 自回归积分滑动平均

### 6. Prophet
- Facebook开源的时序预测模型
- 擅长处理趋势和季节性

## 数据集说明

- **股票池**：上证50成分股（贵州茅台600519、招商银行600036、中国平安601318等）
- **时间范围**：2019-01-01 至今
- **数据频率**：日线数据
- **特征维度**：40+个特征
- **样本构造**：滑动窗口30日，预测下1日
- **数据切分**：
  - 训练集：70%（按时间顺序）
  - 验证集：15%
  - 测试集：15%

## 评估指标

- **Accuracy**：准确率
- **Precision**：精确率
- **Recall**：召回率
- **F1-Score**：F1分数
- **AUC**：ROC曲线下面积
- **Confusion Matrix**：混淆矩阵

## 防止时间泄漏的措施

1. **严格时间切分**：训练集、验证集、测试集严格按时间顺序切分，不打乱
2. **特征计算**：所有技术指标只使用历史数据，不使用当日或未来数据
3. **标签构造**：预测目标为下一交易日涨跌，不使用当日收盘价
4. **滚动验证**：支持时间序列交叉验证

## 配置说明

编辑 `config.yaml` 可以修改：
- 股票代码列表
- 时间窗口大小
- 模型超参数
- 训练参数等

## 注意事项

1. 首次运行会自动下载数据，需要网络连接
2. Mamba模型需要CUDA支持，如无GPU可跳过
3. 数据获取可能受网络影响，建议使用稳定的网络环境
4. 本项目仅供学习研究使用，不构成投资建议

## License

MIT License

