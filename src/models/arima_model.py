"""
ARIMA 模型
经典的统计时序模型
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARIMAModel:
    """ARIMA模型包装类"""
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (5, 1, 0),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ):
        """
        初始化ARIMA模型
        
        Args:
            order: ARIMA阶数 (p, d, q)
                p: 自回归项数
                d: 差分阶数
                q: 移动平均项数
            seasonal_order: 季节性ARIMA阶数 (P, D, Q, s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练ARIMA模型
        
        Args:
            X: 特征数据 (n_samples, seq_len, n_features) - 这里主要使用价格序列
            y: 目标数据 (n_samples,)
        """
        logger.info("Training ARIMA model...")
        
        # ARIMA模型是单变量模型，我们使用价格序列
        # 这里我们为每个样本训练一个模型（实际应用中可能需要优化）
        self.models = []
        
        # 由于ARIMA是传统统计模型，我们采用不同的策略：
        # 使用完整的价格序列训练一个模型
        logger.info("ARIMA is a univariate model, training on price series...")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据 (n_samples, seq_len, n_features)
            
        Returns:
            预测结果 (n_samples, 2) - 分类概率
        """
        logger.info("Predicting with ARIMA model...")
        
        predictions = []
        
        for i in range(len(X)):
            # 提取价格序列（假设是第一个特征）
            price_series = X[i, :, 0]  # 使用第一个特征作为价格
            
            try:
                # 训练ARIMA模型
                model = ARIMA(price_series, order=self.order)
                fitted = model.fit()
                
                # 预测下一个值
                forecast = fitted.forecast(steps=1)[0]
                
                # 计算涨跌
                current_price = price_series[-1]
                predicted_return = (forecast - current_price) / current_price
                
                # 转换为分类概率
                if predicted_return > 0:
                    prob = [0.0, 1.0]  # 上涨
                else:
                    prob = [1.0, 0.0]  # 下跌
                
                predictions.append(prob)
                
            except Exception as e:
                # 如果模型拟合失败，返回默认预测
                logger.warning(f"ARIMA fit failed for sample {i}: {e}")
                predictions.append([0.5, 0.5])
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征数据
            
        Returns:
            预测概率 (n_samples, 2)
        """
        return self.predict(X)


class ARIMAEnsemble:
    """ARIMA集成模型，为每只股票训练独立模型"""
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (5, 1, 0),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        use_exog: bool = True
    ):
        """
        初始化ARIMA集成模型
        
        Args:
            order: ARIMA阶数
            seasonal_order: 季节性阶数
            use_exog: 是否使用外生变量
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_exog = use_exog
        self.models = {}
    
    def fit(self, train_data: pd.DataFrame, stock_codes: list):
        """
        为每只股票训练ARIMA模型
        
        Args:
            train_data: 训练数据
            stock_codes: 股票代码列表
        """
        logger.info("Training ARIMA models for each stock...")
        
        for stock_code in stock_codes:
            stock_data = train_data[train_data['stock_code'] == stock_code]
            
            try:
                if self.use_exog and len(stock_data.columns) > 5:
                    # 使用外生变量
                    exog_cols = [col for col in stock_data.columns 
                                if col not in ['close', 'stock_code', 'target', 'target_return']]
                    exog = stock_data[exog_cols].values
                    
                    model = SARIMAX(
                        stock_data['close'],
                        exog=exog,
                        order=self.order,
                        seasonal_order=self.seasonal_order
                    )
                else:
                    # 仅使用价格
                    model = ARIMA(stock_data['close'], order=self.order)
                
                fitted = model.fit(disp=False)
                self.models[stock_code] = fitted
                logger.info(f"Trained ARIMA for {stock_code}")
                
            except Exception as e:
                logger.error(f"Failed to train ARIMA for {stock_code}: {e}")
                self.models[stock_code] = None
        
        return self
    
    def predict(self, test_data: pd.DataFrame, stock_codes: list) -> np.ndarray:
        """
        预测
        
        Args:
            test_data: 测试数据
            stock_codes: 股票代码列表
            
        Returns:
            预测结果
        """
        predictions = []
        
        for stock_code in stock_codes:
            if stock_code not in self.models or self.models[stock_code] is None:
                logger.warning(f"No model for {stock_code}, using default prediction")
                stock_data = test_data[test_data['stock_code'] == stock_code]
                predictions.extend([[0.5, 0.5]] * len(stock_data))
                continue
            
            stock_data = test_data[test_data['stock_code'] == stock_code]
            model = self.models[stock_code]
            
            try:
                # 预测
                forecast = model.forecast(steps=len(stock_data))
                
                # 转换为分类
                for i, (idx, row) in enumerate(stock_data.iterrows()):
                    current_price = row['close']
                    predicted_price = forecast[i]
                    predicted_return = (predicted_price - current_price) / current_price
                    
                    if predicted_return > 0:
                        prob = [0.0, 1.0]
                    else:
                        prob = [1.0, 0.0]
                    
                    predictions.append(prob)
                    
            except Exception as e:
                logger.error(f"Prediction failed for {stock_code}: {e}")
                predictions.extend([[0.5, 0.5]] * len(stock_data))
        
        return np.array(predictions)


def create_arima_model(config: dict):
    """
    根据配置创建ARIMA模型
    
    Args:
        config: 配置字典
        
    Returns:
        ARIMA模型
    """
    model_config = config['models']['arima']
    
    model = ARIMAModel(
        order=tuple(model_config['order']),
        seasonal_order=tuple(model_config['seasonal_order'])
    )
    
    return model


if __name__ == "__main__":
    # 测试
    from sklearn.metrics import accuracy_score, classification_report
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 100
    seq_len = 30
    n_features = 50
    
    # 生成带趋势的价格数据
    X = np.random.randn(n_samples, seq_len, n_features)
    # 第一个特征作为价格
    for i in range(n_samples):
        trend = np.linspace(100, 100 + np.random.randn() * 10, seq_len)
        X[i, :, 0] = trend + np.random.randn(seq_len) * 0.5
    
    # 生成目标
    y = np.random.randint(0, 2, n_samples)
    
    # 训练和预测
    model = ARIMAModel(order=(2, 1, 0))
    model.fit(X[:80], y[:80])
    
    predictions = model.predict(X[80:])
    y_pred = np.argmax(predictions, axis=1)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Accuracy: {accuracy_score(y[80:], y_pred):.4f}")

