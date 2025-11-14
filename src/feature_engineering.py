"""
特征工程模块
计算各种技术指标，并构造时间序列样本
重点：防止时间数据泄漏
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import logging
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self, config: dict):
        """
        初始化特征工程器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.feature_config = config['features']
        self.indicator_config = self.feature_config['indicators']
        self.scaler = StandardScaler()
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        注意：所有指标只使用历史数据，不使用当日或未来数据
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了技术指标的DataFrame
        """
        logger.info("Calculating technical indicators...")
        
        # 按股票代码分组计算
        df = df.sort_index()
        
        if 'stock_code' in df.columns:
            # 对每只股票单独计算指标
            result_dfs = []
            for stock_code in df['stock_code'].unique():
                stock_df = df[df['stock_code'] == stock_code].copy()
                stock_df = self._calculate_indicators_for_stock(stock_df)
                result_dfs.append(stock_df)
            df = pd.concat(result_dfs, axis=0)
        else:
            df = self._calculate_indicators_for_stock(df)
        
        # 删除NaN值（由于指标计算产生）
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_len - len(df)} rows with NaN after indicator calculation")
        
        logger.info(f"Total features: {len(df.columns)}")
        return df
    
    def _calculate_indicators_for_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        """为单只股票计算技术指标"""
        
        # 1. 基础价格特征
        df = self._calculate_price_features(df)
        
        # 2. 趋势指标
        df = self._calculate_trend_indicators(df)
        
        # 3. 动量指标
        df = self._calculate_momentum_indicators(df)
        
        # 4. 波动率指标
        df = self._calculate_volatility_indicators(df)
        
        # 5. 成交量指标
        df = self._calculate_volume_indicators(df)
        
        # 6. 其他指标
        df = self._calculate_other_indicators(df)
        
        return df
    
    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础价格特征"""
        
        # 涨跌幅
        df['return'] = df['close'].pct_change()
        
        # 对数收益率
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 高低价差
        df['high_low_diff'] = df['high'] - df['low']
        
        # 振幅
        df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1)
        
        # 开盘涨跌幅
        df['open_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # 日内涨跌幅
        df['intraday_return'] = (df['close'] - df['open']) / df['open']
        
        return df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算趋势指标"""
        
        # 简单移动平均线（SMA）
        for period in self.indicator_config['ma_periods']:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            # 价格与均线的偏离度
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
        
        # 指数移动平均线（EMA）
        for period in self.indicator_config['ma_periods']:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1
        
        # MACD
        fast = self.indicator_config['macd_fast']
        slow = self.indicator_config['macd_slow']
        signal = self.indicator_config['macd_signal']
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ADX (Average Directional Index) - 简化版
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': (df['high'] - df['close'].shift(1)).abs(),
            'lc': (df['low'] - df['close'].shift(1)).abs()
        }).max(axis=1)
        
        atr_period = 14
        atr = tr.rolling(window=atr_period).mean()
        plus_di = 100 * (plus_dm.rolling(window=atr_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=atr_period).mean() / atr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=atr_period).mean()
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量指标"""
        
        # RSI (Relative Strength Index)
        rsi_period = self.indicator_config['rsi_period']
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        stoch_period = self.indicator_config['stoch_period']
        low_min = df['low'].rolling(window=stoch_period).min()
        high_max = df['high'].rolling(window=stoch_period).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # CCI (Commodity Channel Index)
        cci_period = self.indicator_config['cci_period']
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(window=cci_period).mean()
        md = tp.rolling(window=cci_period).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (tp - ma) / (0.015 * md)
        
        # Williams %R
        williams_period = self.indicator_config['williams_period']
        high_max = df['high'].rolling(window=williams_period).max()
        low_min = df['low'].rolling(window=williams_period).min()
        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
        
        # ROC (Rate of Change)
        roc_period = self.indicator_config['roc_period']
        df['roc'] = 100 * (df['close'] - df['close'].shift(roc_period)) / df['close'].shift(roc_period)
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        
        return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率指标"""
        
        # Bollinger Bands
        bb_period = self.indicator_config['bb_period']
        bb_std = self.indicator_config['bb_std']
        
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_val = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + bb_std * bb_std_val
        df['bb_lower'] = df['bb_middle'] - bb_std * bb_std_val
        
        # Bollinger Band Width
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Bollinger Band Position
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        atr_period = self.indicator_config['atr_period']
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=atr_period).mean()
        
        # 历史波动率
        df['volatility_10'] = df['return'].rolling(window=10).std()
        df['volatility_20'] = df['return'].rolling(window=20).std()
        df['volatility_30'] = df['return'].rolling(window=30).std()
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量指标"""
        
        # 成交量变化率
        df['volume_change'] = df['volume'].pct_change()
        
        # 成交量移动平均
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        
        # 成交量比率
        df['volume_ratio_5'] = df['volume'] / df['volume_ma_5']
        df['volume_ratio_10'] = df['volume'] / df['volume_ma_10']
        
        # OBV (On-Balance Volume)
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        
        # MFI (Money Flow Index)
        mfi_period = self.indicator_config['mfi_period']
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=mfi_period).sum()
        negative_mf = negative_flow.rolling(window=mfi_period).sum()
        
        mfi_ratio = positive_mf / negative_mf
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        return df
    
    def _calculate_other_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算其他指标"""
        
        # 价格位置（在一定周期内的位置）
        for period in [10, 20, 30]:
            high_max = df['high'].rolling(window=period).max()
            low_min = df['low'].rolling(window=period).min()
            df[f'price_position_{period}'] = (df['close'] - low_min) / (high_max - low_min)
        
        # 距离历史高点
        df['distance_to_high_20'] = (df['close'] - df['high'].rolling(window=20).max()) / df['close']
        df['distance_to_high_60'] = (df['close'] - df['high'].rolling(window=60).max()) / df['close']
        
        # 距离历史低点
        df['distance_to_low_20'] = (df['close'] - df['low'].rolling(window=20).min()) / df['close']
        df['distance_to_low_60'] = (df['close'] - df['low'].rolling(window=60).min()) / df['close']
        
        return df
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建预测目标
        注意：使用shift(-1)获取下一日收盘价，确保不泄漏未来信息
        
        Args:
            df: 包含特征的DataFrame
            
        Returns:
            添加了目标变量的DataFrame
        """
        logger.info("Creating target variable...")
        
        horizon = self.feature_config['prediction_horizon']
        
        # 按股票代码分组处理
        if 'stock_code' in df.columns:
            result_dfs = []
            for stock_code in df['stock_code'].unique():
                stock_df = df[df['stock_code'] == stock_code].copy()
                stock_df = self._create_target_for_stock(stock_df, horizon)
                result_dfs.append(stock_df)
            df = pd.concat(result_dfs, axis=0)
        else:
            df = self._create_target_for_stock(df, horizon)
        
        # 删除最后几行（没有目标值）
        df = df.dropna(subset=['target'])
        
        logger.info(f"Target distribution:\n{df['target'].value_counts()}")
        logger.info(f"Target distribution (%):\n{df['target'].value_counts(normalize=True) * 100}")
        
        return df
    
    def _create_target_for_stock(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """为单只股票创建目标变量"""
        
        # 下一日收盘价
        df['next_close'] = df['close'].shift(-horizon)
        
        # 下一日涨跌幅
        df['next_return'] = (df['next_close'] - df['close']) / df['close']
        
        # 二分类目标：1表示上涨，0表示下跌
        df['target'] = (df['next_return'] > 0).astype(int)
        
        # 也可以保留连续的收益率用于回归任务
        df['target_return'] = df['next_return']
        
        return df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        创建时间序列样本
        使用滑动窗口方法，确保时间顺序
        
        Args:
            df: 包含特征和目标的DataFrame
            sequence_length: 序列长度
            
        Returns:
            X: 特征序列 (n_samples, sequence_length, n_features)
            y: 目标值 (n_samples,)
            feature_names: 特征名称列表
        """
        logger.info("Creating sequences...")
        
        # 选择特征列（排除目标列和辅助列）
        exclude_cols = ['target', 'target_return', 'next_close', 'next_return', 'stock_code']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X_list = []
        y_list = []
        
        # 按股票代码分组处理
        if 'stock_code' in df.columns:
            for stock_code in df['stock_code'].unique():
                stock_df = df[df['stock_code'] == stock_code].copy()
                X_stock, y_stock = self._create_sequences_for_stock(
                    stock_df, feature_cols, sequence_length
                )
                if X_stock is not None:
                    X_list.append(X_stock)
                    y_list.append(y_stock)
        else:
            X_stock, y_stock = self._create_sequences_for_stock(
                df, feature_cols, sequence_length
            )
            if X_stock is not None:
                X_list.append(X_stock)
                y_list.append(y_stock)
        
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        logger.info(f"Sequences shape: X={X.shape}, y={y.shape}")
        logger.info(f"Number of features: {len(feature_cols)}")
        
        return X, y, feature_cols
    
    def _create_sequences_for_stock(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """为单只股票创建序列"""
        
        # 提取特征和目标
        features = df[feature_cols].values
        targets = df['target'].values
        
        if len(features) < sequence_length + 1:
            logger.warning(f"Not enough data points: {len(features)} < {sequence_length + 1}")
            return None, None
        
        X = []
        y = []
        
        # 滑动窗口
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        按时间顺序切分数据集
        重要：不打乱数据，保持时间顺序，防止数据泄漏
        
        Args:
            X: 特征数据
            y: 目标数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        logger.info("Splitting data...")
        
        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        # 按时间顺序切分
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Train target: {np.bincount(y_train)}")
        logger.info(f"Val target: {np.bincount(y_val)}")
        logger.info(f"Test target: {np.bincount(y_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def normalize_features(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        标准化特征
        注意：只使用训练集的统计信息，防止数据泄漏
        
        Args:
            X_train: 训练集特征
            X_val: 验证集特征
            X_test: 测试集特征
            
        Returns:
            标准化后的 X_train, X_val, X_test
        """
        logger.info("Normalizing features...")
        
        # 保存原始形状
        n_train, seq_len, n_features = X_train.shape
        n_val = X_val.shape[0]
        n_test = X_test.shape[0]
        
        # 重塑为2D进行标准化
        X_train_2d = X_train.reshape(-1, n_features)
        X_val_2d = X_val.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)
        
        # 只在训练集上fit
        self.scaler.fit(X_train_2d)
        
        # 转换所有数据集
        X_train_scaled = self.scaler.transform(X_train_2d)
        X_val_scaled = self.scaler.transform(X_val_2d)
        X_test_scaled = self.scaler.transform(X_test_2d)
        
        # 恢复原始形状
        X_train_scaled = X_train_scaled.reshape(n_train, seq_len, n_features)
        X_val_scaled = X_val_scaled.reshape(n_val, seq_len, n_features)
        X_test_scaled = X_test_scaled.reshape(n_test, seq_len, n_features)
        
        logger.info("Normalization completed")
        
        return X_train_scaled, X_val_scaled, X_test_scaled


if __name__ == "__main__":
    # 测试代码
    import yaml
    from data_loader import load_data
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载数据
    df = load_data(config)
    
    # 特征工程
    engineer = FeatureEngineer(config)
    df = engineer.calculate_technical_indicators(df)
    df = engineer.create_target(df)
    
    print(f"\nFeatures shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    
    # 创建序列
    X, y, feature_names = engineer.create_sequences(
        df,
        sequence_length=config['features']['sequence_length']
    )
    
    # 切分数据
    X_train, y_train, X_val, y_val, X_test, y_test = engineer.split_data(
        X, y,
        train_ratio=config['dataset']['train_ratio'],
        val_ratio=config['dataset']['val_ratio']
    )
    
    # 标准化
    X_train, X_val, X_test = engineer.normalize_features(X_train, X_val, X_test)
    
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

