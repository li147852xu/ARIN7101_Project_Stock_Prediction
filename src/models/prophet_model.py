"""
Prophet 模型
Facebook开源的时序预测模型
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProphetModel:
    """Prophet模型包装类"""
    
    def __init__(
        self,
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0
    ):
        """
        初始化Prophet模型
        
        Args:
            yearly_seasonality: 是否考虑年度季节性
            weekly_seasonality: 是否考虑周度季节性
            daily_seasonality: 是否考虑日度季节性
            changepoint_prior_scale: 变点先验尺度
            seasonality_prior_scale: 季节性先验尺度
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.models = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, dates: Optional[pd.DatetimeIndex] = None):
        """
        训练Prophet模型
        
        Args:
            X: 特征数据 (n_samples, seq_len, n_features)
            y: 目标数据 (n_samples,)
            dates: 日期索引
        """
        logger.info("Training Prophet model...")
        
        # Prophet需要特定格式的数据
        # 由于我们的数据是序列格式，需要转换
        logger.info("Prophet model requires time series data with dates")
        
        return self
    
    def predict(self, X: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据 (n_samples, seq_len, n_features)
            dates: 日期索引
            
        Returns:
            预测结果 (n_samples, 2)
        """
        logger.info("Predicting with Prophet model...")
        
        predictions = []
        
        for i in range(len(X)):
            # 提取价格序列
            price_series = X[i, :, 0]
            
            try:
                # 创建Prophet数据格式
                if dates is not None:
                    ds = pd.date_range(end=dates[i], periods=len(price_series), freq='D')
                else:
                    ds = pd.date_range(start='2020-01-01', periods=len(price_series), freq='D')
                
                df = pd.DataFrame({
                    'ds': ds,
                    'y': price_series
                })
                
                # 训练Prophet模型
                model = Prophet(
                    yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    daily_seasonality=self.daily_seasonality,
                    changepoint_prior_scale=self.changepoint_prior_scale,
                    seasonality_prior_scale=self.seasonality_prior_scale
                )
                model.fit(df, verbose=False)
                
                # 预测下一天
                future = model.make_future_dataframe(periods=1)
                forecast = model.predict(future)
                
                # 获取预测值
                predicted_price = forecast['yhat'].iloc[-1]
                current_price = price_series[-1]
                predicted_return = (predicted_price - current_price) / current_price
                
                # 转换为分类概率
                if predicted_return > 0:
                    prob = [0.0, 1.0]
                else:
                    prob = [1.0, 0.0]
                
                predictions.append(prob)
                
            except Exception as e:
                logger.warning(f"Prophet fit failed for sample {i}: {e}")
                predictions.append([0.5, 0.5])
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征数据
            dates: 日期索引
            
        Returns:
            预测概率 (n_samples, 2)
        """
        return self.predict(X, dates)


class ProphetEnsemble:
    """Prophet集成模型，为每只股票训练独立模型"""
    
    def __init__(
        self,
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05
    ):
        """
        初始化Prophet集成模型
        
        Args:
            yearly_seasonality: 年度季节性
            weekly_seasonality: 周度季节性
            daily_seasonality: 日度季节性
            changepoint_prior_scale: 变点先验尺度
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.models = {}
    
    def fit(self, train_data: pd.DataFrame, stock_codes: list):
        """
        为每只股票训练Prophet模型
        
        Args:
            train_data: 训练数据（需要包含日期索引）
            stock_codes: 股票代码列表
        """
        logger.info("Training Prophet models for each stock...")
        
        for stock_code in stock_codes:
            stock_data = train_data[train_data['stock_code'] == stock_code].copy()
            
            try:
                # 准备Prophet格式数据
                df = pd.DataFrame({
                    'ds': stock_data.index,
                    'y': stock_data['close'].values
                })
                
                # 添加额外的回归变量
                if 'volume' in stock_data.columns:
                    df['volume'] = stock_data['volume'].values
                
                # 创建并训练模型
                model = Prophet(
                    yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    daily_seasonality=self.daily_seasonality,
                    changepoint_prior_scale=self.changepoint_prior_scale
                )
                
                # 添加额外回归变量
                if 'volume' in df.columns:
                    model.add_regressor('volume')
                
                model.fit(df, verbose=False)
                self.models[stock_code] = model
                logger.info(f"Trained Prophet for {stock_code}")
                
            except Exception as e:
                logger.error(f"Failed to train Prophet for {stock_code}: {e}")
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
                logger.warning(f"No model for {stock_code}")
                stock_data = test_data[test_data['stock_code'] == stock_code]
                predictions.extend([[0.5, 0.5]] * len(stock_data))
                continue
            
            stock_data = test_data[test_data['stock_code'] == stock_code]
            model = self.models[stock_code]
            
            try:
                # 为每个测试点预测
                for idx, row in stock_data.iterrows():
                    # 创建future dataframe
                    future = pd.DataFrame({
                        'ds': [idx]
                    })
                    
                    if 'volume' in stock_data.columns:
                        future['volume'] = [row['volume']]
                    
                    # 预测
                    forecast = model.predict(future)
                    predicted_price = forecast['yhat'].iloc[0]
                    current_price = row['close']
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


def create_prophet_model(config: dict):
    """
    根据配置创建Prophet模型
    
    Args:
        config: 配置字典
        
    Returns:
        Prophet模型
    """
    model_config = config['models']['prophet']
    
    model = ProphetModel(
        yearly_seasonality=model_config['yearly_seasonality'],
        weekly_seasonality=model_config['weekly_seasonality'],
        daily_seasonality=model_config['daily_seasonality'],
        changepoint_prior_scale=model_config['changepoint_prior_scale']
    )
    
    return model


if __name__ == "__main__":
    # 测试
    from sklearn.metrics import accuracy_score
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 50  # Prophet较慢，使用较少样本
    seq_len = 30
    n_features = 50
    
    # 生成带趋势的价格数据
    X = np.random.randn(n_samples, seq_len, n_features)
    for i in range(n_samples):
        trend = np.linspace(100, 100 + np.random.randn() * 10, seq_len)
        X[i, :, 0] = trend + np.random.randn(seq_len) * 0.5
    
    # 生成目标
    y = np.random.randint(0, 2, n_samples)
    
    # 训练和预测
    model = ProphetModel(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    model.fit(X[:40], y[:40])
    predictions = model.predict(X[40:])
    y_pred = np.argmax(predictions, axis=1)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Accuracy: {accuracy_score(y[40:], y_pred):.4f}")

