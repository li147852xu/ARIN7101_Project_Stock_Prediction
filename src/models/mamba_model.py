"""
Mamba 模型
基于状态空间模型(SSM)的高效序列建模
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not installed. MambaModel will not be available.")


class MambaModel(nn.Module):
    """Mamba模型用于时间序列分类"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        expand: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        初始化Mamba模型
        
        Args:
            input_dim: 输入特征维度
            d_model: 模型维度
            n_layers: Mamba层数
            d_state: 状态维度
            expand: 扩展因子
            num_classes: 分类数量
            dropout: Dropout比率
        """
        super(MambaModel, self).__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is not installed. Please install it with: "
                "pip install mamba-ssm"
            )
        
        self.d_model = d_model
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Mamba层
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                expand=expand
            ) for _ in range(n_layers)
        ])
        
        # Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, seq_len, features)
            
        Returns:
            输出 (batch_size, num_classes)
        """
        # 输入投影
        x = self.input_proj(x)
        
        # 通过Mamba层
        for mamba, norm in zip(self.mamba_layers, self.layer_norms):
            # Mamba层 + 残差连接 + Layer Norm
            residual = x
            x = mamba(x)
            x = self.dropout(x)
            x = norm(x + residual)
        
        # 使用最后一个时间步或全局平均池化
        # x = x.mean(dim=1)  # 全局平均池化
        x = x[:, -1, :]  # 最后一个时间步
        
        # 输出层
        out = self.fc(x)
        
        return out


class SimpleMambaModel(nn.Module):
    """简化的Mamba模型（如果mamba_ssm不可用时的替代）"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        初始化简化Mamba模型
        使用GRU作为替代
        
        Args:
            input_dim: 输入特征维度
            d_model: 模型维度
            n_layers: 层数
            num_classes: 分类数量
            dropout: Dropout比率
        """
        super(SimpleMambaModel, self).__init__()
        
        print("Warning: Using GRU as a fallback for Mamba")
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 使用GRU作为替代
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, seq_len, features)
            
        Returns:
            输出 (batch_size, num_classes)
        """
        # 输入投影
        x = self.input_proj(x)
        
        # GRU
        gru_out, hidden = self.gru(x)
        
        # 使用最后一个隐藏状态
        x = hidden[-1]
        
        # 输出层
        out = self.fc(x)
        
        return out


class HybridMambaModel(nn.Module):
    """混合Mamba模型（结合注意力机制）"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        expand: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        初始化混合Mamba模型
        
        Args:
            input_dim: 输入特征维度
            d_model: 模型维度
            n_layers: Mamba层数
            d_state: 状态维度
            expand: 扩展因子
            num_classes: 分类数量
            dropout: Dropout比率
        """
        super(HybridMambaModel, self).__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm is not installed")
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Mamba层
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                expand=expand
            ) for _ in range(n_layers)
        ])
        
        # Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, seq_len, features)
            
        Returns:
            输出 (batch_size, num_classes)
        """
        # 输入投影
        x = self.input_proj(x)
        
        # 通过Mamba层
        for mamba, norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba(x)
            x = self.dropout(x)
            x = norm(x + residual)
        
        # 注意力层
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 输出层
        out = self.fc(x)
        
        return out


def create_mamba_model(config: dict, input_dim: int, num_classes: int = 2) -> nn.Module:
    """
    根据配置创建Mamba模型
    
    Args:
        config: 配置字典
        input_dim: 输入维度
        num_classes: 类别数
        
    Returns:
        Mamba模型
    """
    model_config = config['models']['mamba']
    
    try:
        model = MambaModel(
            input_dim=input_dim,
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            d_state=model_config['d_state'],
            expand=model_config['expand'],
            num_classes=num_classes,
            dropout=model_config['dropout']
        )
    except (ImportError, KeyError):
        print("Warning: Using SimpleMambaModel (GRU) as fallback")
        model = SimpleMambaModel(
            input_dim=input_dim,
            d_model=model_config.get('d_model', 128),
            n_layers=model_config.get('n_layers', 4),
            num_classes=num_classes,
            dropout=model_config.get('dropout', 0.1)
        )
    
    return model


if __name__ == "__main__":
    # 测试
    batch_size = 32
    seq_len = 30
    features = 50
    
    x = torch.randn(batch_size, seq_len, features)
    
    if MAMBA_AVAILABLE:
        # Mamba模型
        model = MambaModel(
            input_dim=features,
            d_model=128,
            n_layers=4,
            d_state=16,
            expand=2
        )
        out = model(x)
        print(f"Mamba Input shape: {x.shape}")
        print(f"Mamba Output shape: {out.shape}")
        
        # 混合Mamba模型
        model_hybrid = HybridMambaModel(
            input_dim=features,
            d_model=128,
            n_layers=4
        )
        out = model_hybrid(x)
        print(f"\nHybrid Mamba Output shape: {out.shape}")
    else:
        # 简化模型
        model = SimpleMambaModel(
            input_dim=features,
            d_model=128,
            n_layers=4
        )
        out = model(x)
        print(f"Simple Mamba (GRU) Input shape: {x.shape}")
        print(f"Simple Mamba (GRU) Output shape: {out.shape}")

