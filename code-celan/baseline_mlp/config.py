"""Baseline MLP settings."""

from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parent
ADTRANS_ROOT = BASELINE_ROOT.parent / "AdTrans"

TRAINING_CONFIG = {
    "patience": 40,
    "min_delta": 1e-5,
    "max_epochs": 500,
    "device": "cuda",
    "scheduler": {
        "factor": 0.6,
        "patience": 5,
        "min_lr": 1e-8,
    },
    "gradient_clipping": {
        "enabled": True,
        "clip_value": 1.0,
    },
    "l1_regularization": {
        "enabled": True,
        "lambda": 5e-7,
    },
    "model_activation": {
        "type": "GELU",
        "params": {},
    },
}

MODEL_CONFIG = {
    "hidden_layers": 4,
}

OPTIMIZER_CONFIG = {
    "n_trials": 100,
    "seed": 42,
    "dropout_range": (0.1, 0.2),
    "learning_rate_range": (1e-5, 1e-3),
    "weight_decay_range": (1e-8, 1e-6),
    "batch_sizes": [64, 128, 256],
    "hidden_dim_ratio_range": (0.2, 0.8),
}

DATASET_CONFIG = {
    "seed": 42,
    "descriptor_pair": ("skipatom", "magpie"),
    # Subfolder under AdTrans/descriptors/structure_base/ (must match on-disk CSVs).
    "structure_descriptor_name": "chgnet_ft_v_dual",
}

INPUT_CONFIG = {
    "use_encoder": True,
    "use_decoder": True,
    "use_structure": True,
}

RESULTS_DIR = BASELINE_ROOT / "results"

PLOT_CONFIG = {
    "parity_plots_enabled": True,
}


def validate_dataset_config(adtrans_root: Path) -> None:
    """Verify descriptor CSV paths under AdTrans/descriptors."""
    if not INPUT_CONFIG.get("use_structure"):
        return
    name = DATASET_CONFIG.get("structure_descriptor_name")
    if not name:
        raise ValueError(
            "INPUT_CONFIG.use_structure=True requires DATASET_CONFIG.structure_descriptor_name."
        )
    train_csv = adtrans_root / "descriptors" / "structure_base" / name / "train.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(
            f"Structure descriptors not found: {train_csv}\n"
            "Set DATASET_CONFIG.structure_descriptor_name to the folder name under "
            "AdTrans/descriptors/structure_base/."
        )


def validate_input_config(structure_available: bool) -> None:
    enabled = [
        INPUT_CONFIG["use_encoder"],
        INPUT_CONFIG["use_decoder"],
        INPUT_CONFIG["use_structure"],
    ]
    if not any(enabled):
        raise ValueError(
            "INPUT_CONFIG: at least one of use_encoder / use_decoder / use_structure must be True."
        )
    if INPUT_CONFIG["use_structure"] and not structure_available:
        raise ValueError(
            "INPUT_CONFIG.use_structure=True but AdTrans has no structure input "
            "(set structure_placement to encoder, decoder, or all)."
        )


def input_config_tag() -> str:
    parts = []
    if INPUT_CONFIG["use_encoder"]:
        parts.append("enc")
    if INPUT_CONFIG["use_decoder"]:
        parts.append("dec")
    if INPUT_CONFIG["use_structure"]:
        parts.append("struct")
    return "+".join(parts)


def format_input_config_summary() -> str:
    from adtrans_bridge import ADTRANS_DATASET_CONFIG

    enc, dec = DATASET_CONFIG["descriptor_pair"]
    struct_dim = ADTRANS_DATASET_CONFIG.get("structure_descriptor_dim", "?")
    struct_name = DATASET_CONFIG.get("structure_descriptor_name", "?")
    flags = {
        f"encoder ({enc})": INPUT_CONFIG["use_encoder"],
        f"decoder ({dec})": INPUT_CONFIG["use_decoder"],
        f"structure ({struct_name}, {struct_dim}-d)": INPUT_CONFIG["use_structure"],
    }
    active = [name for name, on in flags.items() if on]
    return ", ".join(active) if active else "(none)"
