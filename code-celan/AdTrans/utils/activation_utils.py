import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

def get_activation_module(activation_config: dict) -> nn.Module:

    activation_type = activation_config['type']
    params = activation_config.get('params', {})
    
    if activation_type == 'ReLU':
        return nn.ReLU(**params)
    elif activation_type == 'LeakyReLU':
        return nn.LeakyReLU(**params)
    elif activation_type == 'ELU':
        return nn.ELU(**params)
    elif activation_type == 'Sigmoid':
        return nn.Sigmoid()
    elif activation_type == 'Tanh':
        return nn.Tanh()
    elif activation_type == 'GELU':
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function type: {activation_type}")

def get_functional_activation(activation_config: dict) -> Callable:

    activation_type = activation_config['type']
    if activation_type == 'ReLU':
        return F.relu
    elif activation_type == 'LeakyReLU':
        return F.leaky_relu
    elif activation_type == 'ELU':
        return F.elu
    elif activation_type == 'Sigmoid':
        return torch.sigmoid
    elif activation_type == 'Tanh':
        return torch.tanh
    elif activation_type == 'GELU':
        return F.gelu
    else:
        raise ValueError(f"Unsupported functional activation function type: {activation_type}")
