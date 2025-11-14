"""
LSTM (长短期记忆网络) 模型
经典的时序深度学习模型
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM模型"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        初始化LSTM模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类数量
            dropout: Dropout比率
            bidirectional: 是否使用双向LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 全连接层
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, seq_len, features)
            
        Returns:
            输出 (batch_size, num_classes)
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        if self.bidirectional:
            # 拼接前向和后向的最后隐藏状态
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # 全连接层
        out = self.fc(hidden)
        
        return out


class LSTMAttentionModel(nn.Module):
    """带注意力机制的LSTM模型"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        初始化带注意力的LSTM模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类数量
            dropout: Dropout比率
            bidirectional: 是否使用双向LSTM
        """
        super(LSTMAttentionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1)
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, seq_len, features)
            
        Returns:
            输出 (batch_size, num_classes)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        
        # 注意力机制
        attn_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 加权聚合
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch_size, hidden_dim*2)
        
        # 全连接层
        out = self.fc(context)
        
        return out


class StackedLSTMModel(nn.Module):
    """堆叠LSTM模型，带有残差连接"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        初始化堆叠LSTM模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类数量
            dropout: Dropout比率
        """
        super(StackedLSTMModel, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 多层LSTM
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, seq_len, features)
            
        Returns:
            输出 (batch_size, num_classes)
        """
        # 投影输入
        x = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)
        
        # 逐层处理，带残差连接
        for i, lstm in enumerate(self.lstm_layers):
            lstm_out, _ = lstm(x)
            
            # 残差连接 + LayerNorm
            x = self.layer_norm[i](x + self.dropout(lstm_out))
        
        # 使用最后一个时间步
        x = x[:, -1, :]
        
        # 输出层
        out = self.fc(x)
        
        return out


def create_lstm_model(config: dict, input_dim: int, num_classes: int = 2) -> nn.Module:
    """
    根据配置创建LSTM模型
    
    Args:
        config: 配置字典
        input_dim: 输入维度
        num_classes: 类别数
        
    Returns:
        LSTM模型
    """
    model_config = config['models']['lstm']
    
    # 可以选择使用普通LSTM或带注意力的LSTM
    use_attention = model_config.get('use_attention', False)
    
    if use_attention:
        model = LSTMAttentionModel(
            input_dim=input_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_classes=num_classes,
            dropout=model_config['dropout'],
            bidirectional=model_config['bidirectional']
        )
    else:
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_classes=num_classes,
            dropout=model_config['dropout'],
            bidirectional=model_config['bidirectional']
        )
    
    return model


if __name__ == "__main__":
    # 测试
    batch_size = 32
    seq_len = 30
    features = 50
    
    # 普通LSTM
    model = LSTMModel(input_dim=features, hidden_dim=128, num_layers=2, bidirectional=True)
    x = torch.randn(batch_size, seq_len, features)
    out = model(x)
    print(f"LSTM Input shape: {x.shape}")
    print(f"LSTM Output shape: {out.shape}")
    
    # 带注意力的LSTM
    model_attn = LSTMAttentionModel(input_dim=features, hidden_dim=128, num_layers=2)
    out = model_attn(x)
    print(f"\nLSTM Attention Output shape: {out.shape}")
    
    # 堆叠LSTM
    model_stacked = StackedLSTMModel(input_dim=features, hidden_dim=128, num_layers=3)
    out = model_stacked(x)
    print(f"\nStacked LSTM Output shape: {out.shape}")

