"""Stage-2 decoder token pruning: progressive training driven by XAI rankings."""

PRUNING_STAGE_CONFIG = {
    'enabled': False,
    # Stage 0 uses all descriptor property tokens; each entry drops that many after XAI ranking.
    'prune_per_round': [4, 4, 4, 4],
    # Stop pruning when remaining property tokens reach this floor (>= 1).
    'min_tokens': 1,

    # Pruning token selection uses one XAI CSV (ranking_method).
    # Evaluation still exports IG / FA / self-attention CSVs each round.
    'ranking_method': 'fa',

    'exclude_from_ranking': ['structure'],
    # Always kept during pruning; never dropped by XAI ranking (must fit within target_n_tokens).
    'protected_tokens': [],

    # Token-pruning ranking scope: 'train' or 'all'.
    'xai_ranking_splits': 'train',

    # Optuna: see OPTIMIZER_CONFIG['n_trials'] (stage 0) and ['n_trials_pruning_round'] (later rounds).
    'optuna_visualization': False,

    'hpo_center': {
        'lr_ratio': 10.0,
        'dropout_delta': 0.1,
        'prefer_center_batch': True,
        'enqueue_center_trial': True,
    },

    'seed': 42,
    'device': None,
}
