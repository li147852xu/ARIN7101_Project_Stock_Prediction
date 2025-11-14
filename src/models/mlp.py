"""
MLP (多层感知机) 模型
作为基线模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    """多层感知机模型"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64, 32],
        num_classes: int = 2,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        """
        初始化MLP模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            num_classes: 分类数量
            dropout: Dropout比率
            activation: 激活函数类型
        """
        super(MLPModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 输出层
        self.classifier = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, seq_len, features) 或 (batch_size, features)
            
        Returns:
            输出 (batch_size, num_classes)
        """
        # 如果输入是3D的，展平序列维度
        if len(x.shape) == 3:
            batch_size, seq_len, features = x.shape
            # 使用最后一个时间步或平均池化
            x = x[:, -1, :]  # 使用最后一个时间步
            # 或者使用平均池化: x = x.mean(dim=1)
        
        # 通过特征提取器
        x = self.feature_extractor(x)
        
        # 分类
        out = self.classifier(x)
        
        return out


class MLPModelWithAttention(nn.Module):
    """带有注意力机制的MLP模型"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64, 32],
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        初始化带注意力的MLP模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            num_classes: 分类数量
            dropout: Dropout比率
        """
        super(MLPModelWithAttention, self).__init__()
        
        self.input_dim = input_dim
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
        
        # MLP层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, seq_len, features)
            
        Returns:
            输出 (batch_size, num_classes)
        """
        if len(x.shape) == 2:
            # 如果输入是2D的，添加序列维度
            x = x.unsqueeze(1)
        
        # 计算注意力权重
        batch_size, seq_len, features = x.shape
        attn_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加权聚合
        x = (x * attn_weights).sum(dim=1)  # (batch_size, features)
        
        # MLP
        x = self.mlp(x)
        out = self.classifier(x)
        
        return out


def create_mlp_model(config: dict, input_dim: int, num_classes: int = 2) -> nn.Module:
    """
    根据配置创建MLP模型
    
    Args:
        config: 配置字典
        input_dim: 输入维度
        num_classes: 类别数
        
    Returns:
        MLP模型
    """
    model_config = config['models']['mlp']
    
    model = MLPModel(
        input_dim=input_dim,
        hidden_dims=model_config['hidden_dims'],
        num_classes=num_classes,
        dropout=model_config['dropout'],
        activation=model_config['activation']
    )
    
    return model


if __name__ == "__main__":
    # 测试
    model = MLPModel(input_dim=50, hidden_dims=[128, 64, 32], num_classes=2)
    
    # 3D输入
    x = torch.randn(32, 30, 50)  # (batch, seq_len, features)
    out = model(x)
    print(f"3D Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # 2D输入
    x = torch.randn(32, 50)  # (batch, features)
    out = model(x)
    print(f"\n2D Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # 带注意力的模型
    model_attn = MLPModelWithAttention(input_dim=50, hidden_dims=[128, 64, 32])
    x = torch.randn(32, 30, 50)
    out = model_attn(x)
    print(f"\nAttention MLP Output shape: {out.shape}")

