"""
Transformer 模型
基于注意力机制的时序模型
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        添加位置编码
        
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
            
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer模型用于时间序列分类"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 200
    ):
        """
        初始化Transformer模型
        
        Args:
            input_dim: 输入特征维度
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            num_classes: 分类数量
            dropout: Dropout比率
            max_seq_len: 最大序列长度
        """
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, seq_len, features)
            mask: 注意力掩码
            
        Returns:
            输出 (batch_size, num_classes)
        """
        # 输入投影
        x = self.input_proj(x) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # 全局平均池化或使用最后一个时间步
        # x = x.mean(dim=1)  # 全局平均池化
        x = x[:, -1, :]  # 使用最后一个时间步
        
        # 输出层
        out = self.fc(x)
        
        return out


class TransformerWithCLS(nn.Module):
    """带有CLS token的Transformer模型（类似BERT）"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 200
    ):
        """
        初始化带CLS token的Transformer模型
        
        Args:
            input_dim: 输入特征维度
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            num_classes: 分类数量
            dropout: Dropout比率
            max_seq_len: 最大序列长度
        """
        super(TransformerWithCLS, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len + 1, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, seq_len, features)
            
        Returns:
            输出 (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_proj(x)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 使用CLS token的输出
        cls_output = x[:, 0, :]
        
        # 输出层
        out = self.fc(cls_output)
        
        return out


class TimeSeriesTransformer(nn.Module):
    """专门为时间序列设计的Transformer"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        初始化时间序列Transformer
        
        Args:
            input_dim: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            num_classes: 分类数量
            dropout: Dropout比率
        """
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 使用标准Transformer（编码器-解码器结构）
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
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
        # 投影
        x = self.input_proj(x)
        
        # 使用最后一个时间步作为query
        query = x[:, -1:, :]
        
        # Transformer
        out = self.transformer(x, query)
        
        # 使用解码器输出
        out = out[:, -1, :]
        
        # 分类
        out = self.fc(out)
        
        return out


def create_transformer_model(config: dict, input_dim: int, num_classes: int = 2) -> nn.Module:
    """
    根据配置创建Transformer模型
    
    Args:
        config: 配置字典
        input_dim: 输入维度
        num_classes: 类别数
        
    Returns:
        Transformer模型
    """
    model_config = config['models']['transformer']
    
    model = TransformerModel(
        input_dim=input_dim,
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers'],
        dim_feedforward=model_config['dim_feedforward'],
        num_classes=num_classes,
        dropout=model_config['dropout']
    )
    
    return model


if __name__ == "__main__":
    # 测试
    batch_size = 32
    seq_len = 30
    features = 50
    
    # 标准Transformer
    model = TransformerModel(
        input_dim=features,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=512
    )
    x = torch.randn(batch_size, seq_len, features)
    out = model(x)
    print(f"Transformer Input shape: {x.shape}")
    print(f"Transformer Output shape: {out.shape}")
    
    # 带CLS的Transformer
    model_cls = TransformerWithCLS(
        input_dim=features,
        d_model=128,
        nhead=8,
        num_layers=3
    )
    out = model_cls(x)
    print(f"\nTransformer with CLS Output shape: {out.shape}")
    
    # 时间序列Transformer
    model_ts = TimeSeriesTransformer(
        input_dim=features,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=1
    )
    out = model_ts(x)
    print(f"\nTime Series Transformer Output shape: {out.shape}")

