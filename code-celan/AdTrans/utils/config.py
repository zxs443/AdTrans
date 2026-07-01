TRAINING_CONFIG = {
    'patience': 40,        # Early stopping patience
    'min_delta': 1e-5,     # Minimum improvement threshold
    'max_epochs': 500,     # Maximum number of training epochs
    'device': 'cuda',     # Device selection
    'scheduler': {
        'factor': 0.6,     # Learning rate decay factor
        'patience': 5,     # Learning rate scheduler patience
        'min_lr': 1e-8     # Minimum learning rate
    },
    'model_activation': {
        'type': 'GELU',    # Activation function type for model internals (Transformer layers and output layer)
        'params': {}
    },
    'gradient_clipping': {
        'enabled': True,
        'clip_value': 1.0  # Gradient clipping threshold
    },
    'l1_regularization': {
        'enabled': True,
        'lambda': 5e-7    # L1 regularization strength
    }
}

MODEL_CONFIG = {
    'd_model': 256,             # Transformer internal dimension
    'encoder_nhead': 8,       # Encoder attention heads
    'decoder_nhead': 8,       # Decoder attention heads
    'pooling_nhead': 4,       # Attention pooling layer attention heads
    'num_layers': 2,          # Transformer encoder and decoder layers
    'output_hidden_layers': 2, # Number of hidden layers in output layer
}

OPTIMIZER_CONFIG = {
    'n_trials': 100,        # Stage 0 (full tokens) & single-run main.py when pruning disabled
    'n_trials_pruning_round': 100,  # Pruning rounds after stage 0 (int or list per round)
    'seed': 42,
    'dropout_range': (0.1, 0.2),  # Dropout rate range
    'learning_rate_range': (1e-5, 1e-3),  # Learning rate range
    'weight_decay_range': (1e-8, 1e-6),  # Weight decay range (L2)
    'batch_sizes': [64, 128, 256],  # Batch size options
    'output_dim_ratio_range': (0.2, 0.8), # Ratio range for output layer hidden dimensions relative to d_model
     
    'visualization': {
        'enabled': True,
        'plots_to_generate': ['history', 'parallel_coordinate', 'slice', 'param_importances']
    }
}

XAI_CONFIG = {
    'integrated_gradients': {
        'enabled': True,
        'steps': 30, # Number of integration steps
        'baseline_type': 'zeros', # 'zeros' or 'mean' (mean uses zero baseline on standardized inputs)
        'visualize_top_k_properties': 15,
    },
    'feature_ablation': {
        'enabled': True,
        'ablate_elements': True, # Whether to ablate elements
        'ablate_properties': True, # Whether to ablate property tokens
        'ablation_value': 'zeros', # 'zeros' or 'random' baseline value type
        'visualize_top_k_properties': 15,
    }
}

DATASET_CONFIG = {
    'seed': 42,
    'structure_descriptor_name': 'chgnet_ft_v_dual',
    # skipatom/mat2vec: 200 element dims + 20 Z enc; magpie encoder would be 42 (=22+20)
    'encoder_token_dim': 220,
    'decoder_token_dim': 16,  # Feature width per decoder property token (Magpie stats × methods per token)
    'use_causal_mask_decoder': False, # Whether to use causal mask in decoder self-attention, in this case, we do not use it
    'descriptor_pair': ('skipatom', 'magpie'),  # encoder-decoder descriptor pair
    # Structure placement: 'none' | 'encoder' | 'decoder' | 'all'
    # encoder: structure token(s) before element slots
    # decoder: structure token(s) after property tokens
    # all: both sides; dual split -> 2 tokens per side (charge, discharge)
    # In this project, always keep the setting set to “all.”
    'structure_placement': 'all',
    # CSV flat width: 64 discharge-only, or 128 charge|discharge (finetune concat layout)
    'structure_descriptor_dim': 128,
    # True: reshape 128-d -> 2x64 tokens (recommended for 128-d concat CSVs)
    # False: one token with full flat dim (128 -> single structure embed)
    # In this proj, keep False all the time.
    'structure_dual_token_split': False,
}
