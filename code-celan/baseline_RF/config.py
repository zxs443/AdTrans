"""Baseline Random Forest settings (mirrors baseline_mlp conventions)."""

from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parent
ADTRANS_ROOT = BASELINE_ROOT.parent / "AdTrans"

MODEL_CONFIG = {
    # Fixed Random Forest settings (not searched by Optuna).
    "bootstrap": True,
    "n_jobs": -1,          # Use all CPU cores for tree fitting.
    "criterion": "squared_error",
}

OPTIMIZER_CONFIG = {
    # Same trial budget / seed / objective (val voltage MSE) as the MLP baseline.
    "n_trials": 100,
    "seed": 42,
    # Random Forest hyperparameter search space.
    "n_estimators_range": (200, 1000),      # Number of trees
    "max_depth_range": (8, 64),             # Log-uniform; 64 is effectively unlimited here
    "min_samples_split_range": (2, 20),
    "min_samples_leaf_range": (1, 10),
    "max_features_range": (0.1, 1.0),       # Fraction of features considered per split
    "max_samples_range": (0.5, 1.0),        # Bootstrap subsample fraction per tree
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
    # Fitted forests can be several hundred MB; disable if disk space matters.
    "save_model_enabled": True,
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
