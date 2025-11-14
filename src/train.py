"""
模型训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: str = 'cuda'
    ):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            config: 配置字典
            device: 设备（cuda或cpu）
        """
        self.model = model
        self.config = config
        self.train_config = config['training']
        
        # 设置设备
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            device = 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 损失函数
        self.criterion = self._create_criterion()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # 早停
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_name = self.train_config['optimizer'].lower()
        lr = self.train_config['learning_rate']
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            return optim.Adam(self.model.parameters(), lr=lr)
    
    def _create_criterion(self) -> nn.Module:
        """创建损失函数"""
        loss_name = self.train_config['loss'].lower()
        
        if loss_name == 'cross_entropy':
            # 处理类别不平衡
            if self.train_config.get('use_class_weights', False):
                # 权重将在训练时计算
                return nn.CrossEntropyLoss()
            else:
                return nn.CrossEntropyLoss()
        elif loss_name == 'focal_loss':
            return FocalLoss()
        else:
            return nn.CrossEntropyLoss()
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_config = self.train_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config.get('patience', 10),
                factor=scheduler_config.get('factor', 0.5)
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_config['epochs']
            )
        else:
            return None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            训练历史
        """
        logger.info("Starting training...")
        
        # 创建数据加载器
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
        
        # 训练循环
        for epoch in range(self.train_config['epochs']):
            # 训练
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self._validate(val_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 日志
            logger.info(
                f"Epoch {epoch+1}/{self.train_config['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 早停
            if self._early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training completed")
        return self.history
    
    def _train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = False
    ) -> DataLoader:
        """创建数据加载器"""
        # 转换为张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # 创建数据集
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=shuffle,
            num_workers=0,  # 在Windows上设置为0
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return dataloader
    
    def _early_stopping(self, val_loss: float) -> bool:
        """早停检查"""
        early_stop_config = self.train_config.get('early_stopping', {})
        patience = early_stop_config.get('patience', 20)
        min_delta = early_stop_config.get('min_delta', 0.001)
        
        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                return True
        
        return False
    
    def save_model(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Model loaded from {path}")


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        初始化Focal Loss
        
        Args:
            alpha: 平衡因子
            gamma: 聚焦参数
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        Args:
            inputs: 模型输出 (batch_size, num_classes)
            targets: 目标标签 (batch_size,)
            
        Returns:
            损失值
        """
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    model_name: str = "model"
) -> Trainer:
    """
    训练模型的便捷函数
    
    Args:
        model: 要训练的模型
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        config: 配置字典
        model_name: 模型名称
        
    Returns:
        训练器对象
    """
    # 创建训练器
    device = config['training']['device']
    trainer = Trainer(model, config, device=device)
    
    # 训练
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # 保存模型
    save_path = f"results/models/{model_name}.pth"
    trainer.save_model(save_path)
    
    return trainer


if __name__ == "__main__":
    # 测试
    from models.mlp import MLPModel
    import yaml
    
    # 加载配置
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建测试数据
    np.random.seed(42)
    n_train = 1000
    n_val = 200
    seq_len = 30
    n_features = 50
    
    X_train = np.random.randn(n_train, seq_len, n_features)
    y_train = np.random.randint(0, 2, n_train)
    X_val = np.random.randn(n_val, seq_len, n_features)
    y_val = np.random.randint(0, 2, n_val)
    
    # 创建模型
    model = MLPModel(input_dim=n_features)
    
    # 训练
    config['training']['epochs'] = 5  # 快速测试
    trainer = train_model(model, X_train, y_train, X_val, y_val, config, "test_mlp")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")

