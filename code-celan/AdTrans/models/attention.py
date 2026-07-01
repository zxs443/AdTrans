import torch
import torch.nn as nn
from utils.config import TRAINING_CONFIG 
from utils.activation_utils import get_activation_module 

class AttentionPooling(nn.Module):
    """Single-query attention pooling over decoder token sequence."""
    
    def __init__(self, d_model, nhead=4, dropout_rate=0.2, activation=None): 
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads=nhead, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.activation = activation if activation is not None else \
                          get_activation_module(TRAINING_CONFIG['model_activation'])

    def forward(self, x):  
        batch_size = x.size(0)
        q = self.query.expand(batch_size, -1, -1)  
        out, attn_weights = self.attn(q, x, x, need_weights=not self.training) 
        out = self.activation(out)
        out = self.dropout(out)
        return out.squeeze(1), attn_weights 
