"""
数据获取模块
支持从多个数据源获取股票数据
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataLoader:
    """股票数据加载器"""
    
    def __init__(self, data_source: str = "yfinance", data_dir: str = "data/raw"):
        """
        初始化数据加载器
        
        Args:
            data_source: 数据源，支持 'yfinance' 或 'akshare'
            data_dir: 数据保存目录
        """
        self.data_source = data_source
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_stock_data(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        下载股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'，None表示到最新
            force_download: 是否强制重新下载
            
        Returns:
            合并后的股票数据DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        all_data = []
        
        for stock_code in stock_codes:
            logger.info(f"Downloading data for {stock_code}...")
            
            # 检查本地缓存
            cache_file = os.path.join(
                self.data_dir, 
                f"{stock_code.replace('.SS', '')}_{start_date}_{end_date}.csv"
            )
            
            if os.path.exists(cache_file) and not force_download:
                logger.info(f"Loading from cache: {cache_file}")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            else:
                # 从数据源下载
                if self.data_source == "yfinance":
                    df = self._download_from_yfinance(stock_code, start_date, end_date)
                elif self.data_source == "akshare":
                    df = self._download_from_akshare(stock_code, start_date, end_date)
                else:
                    raise ValueError(f"Unsupported data source: {self.data_source}")
                
                # 保存到本地
                if df is not None and not df.empty:
                    df.to_csv(cache_file)
                    logger.info(f"Saved to cache: {cache_file}")
            
            if df is not None and not df.empty:
                df['stock_code'] = stock_code
                all_data.append(df)
            else:
                logger.warning(f"No data downloaded for {stock_code}")
        
        if not all_data:
            raise ValueError("No data downloaded for any stock")
        
        # 合并所有股票数据
        combined_df = pd.concat(all_data, axis=0)
        combined_df.sort_index(inplace=True)
        
        logger.info(f"Total data points: {len(combined_df)}")
        return combined_df
    
    def _download_from_yfinance(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """从yfinance下载数据"""
        try:
            import yfinance as yf
            
            stock = yf.Ticker(stock_code)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data returned for {stock_code}")
                return None
            
            # 标准化列名
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # 只保留需要的列
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading from yfinance: {e}")
            return None
    
    def _download_from_akshare(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """从akshare下载数据"""
        try:
            import akshare as ak
            
            # akshare使用6位股票代码
            code = stock_code.replace('.SS', '').replace('.SZ', '')
            
            # 获取日线数据
            df = ak.stock_zh_a_hist(
                symbol=code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"  # 前复权
            )
            
            if df.empty:
                logger.warning(f"No data returned for {stock_code}")
                return None
            
            # 标准化列名
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume'
            })
            
            # 设置日期为索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 只保留需要的列
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading from akshare: {e}")
            return None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        Args:
            df: 原始数据
            
        Returns:
            清洗后的数据
        """
        logger.info("Cleaning data...")
        
        # 删除缺失值
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_len - len(df)} rows with missing values")
        
        # 删除重复行
        initial_len = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_len - len(df)} duplicate rows")
        
        # 删除异常值（价格为0或负数）
        for col in ['open', 'high', 'low', 'close']:
            initial_len = len(df)
            df = df[df[col] > 0]
            if initial_len > len(df):
                logger.info(f"Removed {initial_len - len(df)} rows with invalid {col}")
        
        # 删除成交量为0的行
        initial_len = len(df)
        df = df[df['volume'] > 0]
        logger.info(f"Removed {initial_len - len(df)} rows with zero volume")
        
        # 确保 high >= low
        initial_len = len(df)
        df = df[df['high'] >= df['low']]
        logger.info(f"Removed {initial_len - len(df)} rows where high < low")
        
        # 确保 high >= close >= low
        initial_len = len(df)
        df = df[(df['close'] <= df['high']) & (df['close'] >= df['low'])]
        logger.info(f"Removed {initial_len - len(df)} rows where close out of range")
        
        logger.info(f"Final data points: {len(df)}")
        return df


def load_data(config: dict) -> pd.DataFrame:
    """
    根据配置加载数据
    
    Args:
        config: 配置字典
        
    Returns:
        股票数据DataFrame
    """
    data_config = config['data']
    
    loader = StockDataLoader(
        data_source=data_config['data_source'],
        data_dir=data_config['raw_data_dir']
    )
    
    # 下载数据
    df = loader.download_stock_data(
        stock_codes=data_config['stock_codes'],
        start_date=data_config['start_date'],
        end_date=data_config['end_date']
    )
    
    # 清洗数据
    df = loader.clean_data(df)
    
    return df


if __name__ == "__main__":
    # 测试代码
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    df = load_data(config)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

