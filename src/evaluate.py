"""
模型评估脚本
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """模型评估器"""
    
    def __init__(self, config: dict):
        """
        初始化评估器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.eval_config = config['evaluation']
        self.results_dir = self.eval_config['results_dir']
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 可视化配置
        self.viz_config = config.get('visualization', {})
        if self.viz_config.get('enable', True):
            self.plots_dir = self.viz_config.get('plots_dir', 'plots')
            os.makedirs(self.plots_dir, exist_ok=True)
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        device: str = 'cuda'
    ) -> Dict:
        """
        评估模型
        
        Args:
            model: 要评估的模型
            X_test: 测试特征
            y_test: 测试标签
            model_name: 模型名称
            device: 设备
            
        Returns:
            评估指标字典
        """
        logger.info(f"Evaluating {model_name}...")
        
        # 获取预测
        if isinstance(model, nn.Module):
            y_pred, y_pred_proba = self._predict_torch_model(model, X_test, device)
        else:
            # 对于sklearn风格的模型（ARIMA, Prophet）
            y_pred_proba = model.predict_proba(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 计算指标
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # 打印结果
        self._print_metrics(metrics, model_name)
        
        # 保存混淆矩阵
        if self.eval_config.get('save_confusion_matrix', True):
            self._plot_confusion_matrix(
                y_test, y_pred, model_name
            )
        
        # 保存ROC曲线
        if self.eval_config.get('save_roc_curve', True):
            self._plot_roc_curve(
                y_test, y_pred_proba, model_name
            )
        
        # 保存详细报告
        self._save_classification_report(y_test, y_pred, model_name)
        
        return metrics
    
    def _predict_torch_model(
        self,
        model: nn.Module,
        X: np.ndarray,
        device: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用PyTorch模型预测
        
        Args:
            model: PyTorch模型
            X: 输入特征
            device: 设备
            
        Returns:
            预测标签和概率
        """
        model.eval()
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 批量预测
        batch_size = 256
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = torch.FloatTensor(X[i:i+batch_size]).to(device)
                outputs = model(batch_X)
                
                # 获取概率
                probs = torch.softmax(outputs, dim=1)
                
                # 获取预测
                _, preds = outputs.max(1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """
        计算评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 基本指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1'] = f1_score(y_true, y_pred, average='binary')
        
        # AUC
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except Exception as e:
            logger.warning(f"Failed to calculate AUC: {e}")
            metrics['auc'] = 0.0
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # 每个类别的指标
        metrics['precision_per_class'] = precision_score(
            y_true, y_pred, average=None
        )
        metrics['recall_per_class'] = recall_score(
            y_true, y_pred, average=None
        )
        metrics['f1_per_class'] = f1_score(
            y_true, y_pred, average=None
        )
        
        return metrics
    
    def _print_metrics(self, metrics: Dict, model_name: str):
        """打印评估指标"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Results for {model_name}")
        logger.info(f"{'='*50}")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"AUC:       {metrics['auc']:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"\n{metrics['confusion_matrix']}")
        logger.info(f"{'='*50}\n")
    
    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Down', 'Up'],
            yticklabels=['Down', 'Up']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        save_path = os.path.join(self.plots_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(save_path, dpi=self.viz_config.get('dpi', 300), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def _plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str
    ):
        """绘制ROC曲线"""
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc='lower right')
            plt.grid(alpha=0.3)
            
            save_path = os.path.join(self.plots_dir, f'{model_name}_roc_curve.png')
            plt.savefig(save_path, dpi=self.viz_config.get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            logger.info(f"ROC curve saved to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to plot ROC curve: {e}")
    
    def _save_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ):
        """保存分类报告"""
        report = classification_report(
            y_true, y_pred,
            target_names=['Down', 'Up'],
            digits=4
        )
        
        save_path = os.path.join(
            self.results_dir,
            f'{model_name}_classification_report.txt'
        )
        
        with open(save_path, 'w') as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("="*50 + "\n\n")
            f.write(report)
        
        logger.info(f"Classification report saved to {save_path}")
    
    def compare_models(self, results: Dict[str, Dict]):
        """
        比较多个模型的性能
        
        Args:
            results: 模型名称到指标字典的映射
        """
        logger.info("\nComparing models...")
        
        # 创建比较表
        comparison_df = pd.DataFrame({
            model_name: {
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'AUC': metrics['auc']
            }
            for model_name, metrics in results.items()
        }).T
        
        # 打印比较结果
        logger.info("\n" + "="*70)
        logger.info("Model Comparison")
        logger.info("="*70)
        logger.info("\n" + comparison_df.to_string())
        logger.info("\n" + "="*70)
        
        # 保存比较结果
        save_path = os.path.join(self.results_dir, 'model_comparison.csv')
        comparison_df.to_csv(save_path)
        logger.info(f"\nModel comparison saved to {save_path}")
        
        # 绘制比较图
        self._plot_model_comparison(comparison_df)
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame):
        """绘制模型比较图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            comparison_df[metric].plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # 删除多余的子图
        fig.delaxes(axes[-1])
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=self.viz_config.get('dpi', 300), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to {save_path}")
    
    def plot_training_history(self, history: Dict, model_name: str):
        """
        绘制训练历史
        
        Args:
            history: 训练历史字典
            model_name: 模型名称
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_name} - Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(history['train_acc'], label='Train Accuracy')
        axes[1].plot(history['val_acc'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{model_name} - Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'{model_name}_training_history.png')
        plt.savefig(save_path, dpi=self.viz_config.get('dpi', 300), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved to {save_path}")


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    model_name: str = "model",
    device: str = 'cuda'
) -> Dict:
    """
    评估模型的便捷函数
    
    Args:
        model: 要评估的模型
        X_test: 测试特征
        y_test: 测试标签
        config: 配置字典
        model_name: 模型名称
        device: 设备
        
    Returns:
        评估指标字典
    """
    evaluator = Evaluator(config)
    metrics = evaluator.evaluate_model(model, X_test, y_test, model_name, device)
    return metrics


if __name__ == "__main__":
    # 测试
    import yaml
    from models.mlp import MLPModel
    
    # 加载配置
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建测试数据
    np.random.seed(42)
    n_test = 500
    seq_len = 30
    n_features = 50
    
    X_test = np.random.randn(n_test, seq_len, n_features)
    y_test = np.random.randint(0, 2, n_test)
    
    # 创建模型
    model = MLPModel(input_dim=n_features)
    
    # 评估
    metrics = evaluate_model(model, X_test, y_test, config, "test_mlp", device='cpu')
    
    print("\nEvaluation completed!")

