"""
主执行脚本
股票短期涨跌预测完整流程
"""

import os
import sys
import yaml
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_data
from src.feature_engineering import FeatureEngineer
from src.train import train_model
from src.evaluate import Evaluator

# 模型导入
from src.models.mlp import create_mlp_model
from src.models.lstm import create_lstm_model
from src.models.transformer import create_transformer_model
from src.models.mamba_model import create_mamba_model
from src.models.arima_model import create_arima_model
from src.models.prophet_model import create_prophet_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories(config: dict):
    """创建必要的目录"""
    dirs = [
        config['data']['raw_data_dir'],
        config['data']['processed_data_dir'],
        config['evaluation']['results_dir'],
        config['visualization']['plots_dir'],
        config['logging']['log_dir'],
        'results/models'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    logger.info("Directories created successfully")


def load_and_prepare_data(config: dict):
    """加载并准备数据"""
    logger.info("="*70)
    logger.info("Step 1: Loading and preparing data")
    logger.info("="*70)
    
    # 加载数据
    df = load_data(config)
    logger.info(f"Loaded {len(df)} data points")
    
    # 特征工程
    engineer = FeatureEngineer(config)
    
    # 计算技术指标
    df = engineer.calculate_technical_indicators(df)
    logger.info(f"Calculated technical indicators, shape: {df.shape}")
    
    # 创建目标变量
    df = engineer.create_target(df)
    logger.info("Created target variable")
    
    # 保存处理后的数据
    processed_path = os.path.join(
        config['data']['processed_data_dir'],
        'processed_data.csv'
    )
    df.to_csv(processed_path)
    logger.info(f"Saved processed data to {processed_path}")
    
    # 创建序列样本
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
    
    # 标准化特征
    X_train, X_val, X_test = engineer.normalize_features(X_train, X_val, X_test)
    
    logger.info(f"Data preparation completed")
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def train_all_models(
    X_train, y_train, X_val, y_val,
    config: dict,
    models_to_train: list = None
):
    """训练所有模型"""
    logger.info("\n" + "="*70)
    logger.info("Step 2: Training models")
    logger.info("="*70)
    
    input_dim = X_train.shape[2]
    trained_models = {}
    
    # 默认训练所有模型
    if models_to_train is None:
        models_to_train = ['mlp', 'lstm', 'transformer', 'mamba']
    
    # MLP
    if 'mlp' in models_to_train:
        logger.info("\n--- Training MLP ---")
        try:
            mlp_model = create_mlp_model(config, input_dim)
            mlp_trainer = train_model(
                mlp_model, X_train, y_train, X_val, y_val,
                config, model_name='mlp'
            )
            trained_models['MLP'] = {
                'model': mlp_model,
                'trainer': mlp_trainer,
                'type': 'torch'
            }
            logger.info("MLP training completed")
        except Exception as e:
            logger.error(f"MLP training failed: {e}")
    
    # LSTM
    if 'lstm' in models_to_train:
        logger.info("\n--- Training LSTM ---")
        try:
            lstm_model = create_lstm_model(config, input_dim)
            lstm_trainer = train_model(
                lstm_model, X_train, y_train, X_val, y_val,
                config, model_name='lstm'
            )
            trained_models['LSTM'] = {
                'model': lstm_model,
                'trainer': lstm_trainer,
                'type': 'torch'
            }
            logger.info("LSTM training completed")
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
    
    # Transformer
    if 'transformer' in models_to_train:
        logger.info("\n--- Training Transformer ---")
        try:
            transformer_model = create_transformer_model(config, input_dim)
            transformer_trainer = train_model(
                transformer_model, X_train, y_train, X_val, y_val,
                config, model_name='transformer'
            )
            trained_models['Transformer'] = {
                'model': transformer_model,
                'trainer': transformer_trainer,
                'type': 'torch'
            }
            logger.info("Transformer training completed")
        except Exception as e:
            logger.error(f"Transformer training failed: {e}")
    
    # Mamba
    if 'mamba' in models_to_train:
        logger.info("\n--- Training Mamba ---")
        try:
            mamba_model = create_mamba_model(config, input_dim)
            mamba_trainer = train_model(
                mamba_model, X_train, y_train, X_val, y_val,
                config, model_name='mamba'
            )
            trained_models['Mamba'] = {
                'model': mamba_model,
                'trainer': mamba_trainer,
                'type': 'torch'
            }
            logger.info("Mamba training completed")
        except Exception as e:
            logger.error(f"Mamba training failed: {e}")
    
    # ARIMA (统计模型，训练方式不同)
    if 'arima' in models_to_train:
        logger.info("\n--- Training ARIMA ---")
        try:
            arima_model = create_arima_model(config)
            arima_model.fit(X_train, y_train)
            trained_models['ARIMA'] = {
                'model': arima_model,
                'trainer': None,
                'type': 'statistical'
            }
            logger.info("ARIMA training completed")
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
    
    # Prophet
    if 'prophet' in models_to_train:
        logger.info("\n--- Training Prophet ---")
        try:
            prophet_model = create_prophet_model(config)
            prophet_model.fit(X_train, y_train)
            trained_models['Prophet'] = {
                'model': prophet_model,
                'trainer': None,
                'type': 'statistical'
            }
            logger.info("Prophet training completed")
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
    
    return trained_models


def evaluate_all_models(
    trained_models: dict,
    X_test, y_test,
    config: dict
):
    """评估所有模型"""
    logger.info("\n" + "="*70)
    logger.info("Step 3: Evaluating models")
    logger.info("="*70)
    
    evaluator = Evaluator(config)
    results = {}
    
    device = config['training']['device']
    
    for model_name, model_info in trained_models.items():
        logger.info(f"\n--- Evaluating {model_name} ---")
        try:
            model = model_info['model']
            trainer = model_info['trainer']
            
            # 评估模型
            metrics = evaluator.evaluate_model(
                model, X_test, y_test, model_name, device
            )
            results[model_name] = metrics
            
            # 绘制训练历史（如果有）
            if trainer is not None and hasattr(trainer, 'history'):
                evaluator.plot_training_history(trainer.history, model_name)
        
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
    
    # 比较所有模型
    if results:
        evaluator.compare_models(results)
    
    return results


def main(args):
    """主函数"""
    logger.info("="*70)
    logger.info("Stock Price Movement Prediction")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    seed = config['training']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 创建目录
    setup_directories(config)
    
    # 根据步骤执行
    if args.step == 'all' or args.step == 'download':
        # 加载和准备数据
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = \
            load_and_prepare_data(config)
        
        # 保存数据供后续使用
        np.savez(
            'data/processed/datasets.npz',
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )
        logger.info("Saved datasets to data/processed/datasets.npz")
        
        if args.step == 'download':
            return
    
    # 加载已保存的数据
    if args.step in ['train', 'evaluate']:
        logger.info("Loading saved datasets...")
        data = np.load('data/processed/datasets.npz')
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
    
    if args.step == 'all' or args.step == 'train':
        # 训练模型
        models_to_train = args.model.split(',') if args.model else None
        trained_models = train_all_models(
            X_train, y_train, X_val, y_val,
            config, models_to_train
        )
        
        if args.step == 'train':
            return
    
    if args.step == 'all' or args.step == 'evaluate':
        # 如果只运行评估，需要加载已训练的模型
        if args.step == 'evaluate':
            trained_models = {}
            
            # 加载PyTorch模型
            for model_name in ['mlp', 'lstm', 'transformer', 'mamba']:
                model_path = f'results/models/{model_name}.pth'
                if os.path.exists(model_path):
                    try:
                        logger.info(f"Loading {model_name} from {model_path}")
                        
                        input_dim = X_train.shape[2]
                        
                        if model_name == 'mlp':
                            model = create_mlp_model(config, input_dim)
                        elif model_name == 'lstm':
                            model = create_lstm_model(config, input_dim)
                        elif model_name == 'transformer':
                            model = create_transformer_model(config, input_dim)
                        elif model_name == 'mamba':
                            model = create_mamba_model(config, input_dim)
                        
                        # 加载权重
                        checkpoint = torch.load(model_path, map_location='cpu')
                        model.load_state_dict(checkpoint['model_state_dict'])
                        
                        trained_models[model_name.upper()] = {
                            'model': model,
                            'trainer': None,
                            'type': 'torch'
                        }
                    except Exception as e:
                        logger.error(f"Failed to load {model_name}: {e}")
        
        # 评估所有模型
        results = evaluate_all_models(trained_models, X_test, y_test, config)
    
    logger.info("\n" + "="*70)
    logger.info("All tasks completed!")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Stock Price Movement Prediction'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--step',
        type=str,
        default='all',
        choices=['all', 'download', 'train', 'evaluate'],
        help='Which step to run'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Models to train (comma-separated), e.g., mlp,lstm,transformer'
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        sys.exit(1)

