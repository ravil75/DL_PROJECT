"""
Архитектура Probe модели
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustProbe(nn.Module):
    """
    Probe с Transformer encoder и multi-layer fusion.
    
    Особенности:
    - Learnable layer weights для комбинации слоёв
    - Attention pooling
    - Сильная регуляризация (dropout, noise)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_extra_layers: int,
        num_heads: int = 4,
        num_transformer_layers: int = 1,
        ff_dim: int = 256,
        max_seq_len: int = 200,
        dropout: float = 0.4,
        noise_std: float = 0.02
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_extra_layers = num_extra_layers
        self.dropout_rate = dropout
        self.noise_std = noise_std
        
        # Learnable layer weights
        self.layer_weights = nn.Parameter(
            torch.ones(num_extra_layers + 1) / (num_extra_layers + 1)
        )
        
        # Input processing
        self.input_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Positional encoding (frozen sinusoidal)
        self.register_buffer(
            'pos_encoding', 
            self._get_sinusoidal_encoding(max_seq_len, hidden_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # Attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=2, dropout=dropout, batch_first=True
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)
        
        # Extra layers processing
        self.extra_norm = nn.LayerNorm(hidden_dim)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        self._init_weights()
    
    def _get_sinusoidal_encoding(self, max_seq_len: int, d_model: int) -> torch.Tensor:
        """Синусоидальное позиционное кодирование"""
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pos_encoding = torch.zeros(max_seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)
    
    def _init_weights(self):
        """Инициализация весов"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        main_seq: torch.Tensor,
        main_mask: torch.Tensor,
        extra_pooled: list,
        add_noise: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            main_seq: (batch, seq_len, hidden_dim) - основные hidden states
            main_mask: (batch, seq_len) - маска внимания
            extra_pooled: list of (batch, hidden_dim) - pooled features из extra слоёв
            add_noise: добавлять ли шум (только при обучении)
            
        Returns:
            logits: (batch, 2)
        """
        batch_size, seq_len, _ = main_seq.shape
        
        # Data augmentation (только при обучении)
        if self.training:
            main_seq = self.input_dropout(main_seq)
            if add_noise and self.noise_std > 0:
                main_seq = main_seq + torch.randn_like(main_seq) * self.noise_std
        
        # Process main sequence
        x = self.input_norm(main_seq)
        x = x + 0.1 * self.pos_encoding[:, :seq_len, :]
        
        # Transformer
        mask = (main_mask == 0)
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Attention pooling
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attention(query, x, x, key_padding_mask=mask)
        main_features = self.pool_norm(pooled.squeeze(1))
        
        # Process extra layers with learned weights
        weights = F.softmax(self.layer_weights, dim=0)
        extra_stack = torch.stack(extra_pooled, dim=0)
        weighted_extra = torch.einsum('l,lbh->bh', weights[1:], extra_stack)
        weighted_main = weights[0] * main_features
        extra_features = self.extra_norm(weighted_extra + weighted_main)
        
        # Fusion
        combined = torch.cat([main_features, extra_features], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
    
    @property
    def num_parameters(self) -> int:
        """Количество параметров модели"""
        return sum(p.numel() for p in self.parameters())