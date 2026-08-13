"""Flattened Random Forest baseline on AdTrans descriptors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASELINE_ROOT = Path(__file__).resolve().parent
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

import adtrans_bridge  
from stdio_utils import configure_realtime_stdio

configure_realtime_stdio()

from config import (
    ADTRANS_ROOT,
    DATASET_CONFIG,
    OPTIMIZER_CONFIG,
    format_input_config_summary,
    input_config_tag,
)
from train import run_baseline


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def main() -> None:
    seed = DATASET_CONFIG["seed"]
    set_seed(seed)

    enc, dec = DATASET_CONFIG["descriptor_pair"]
    print(
        f"Random Forest baseline | seed={seed} | "
        f"pair=({enc}, {dec}) | inputs={input_config_tag()}",
        flush=True,
    )
    print(f"Descriptors: {ADTRANS_ROOT / 'descriptors'}", flush=True)
    print(f"Flat blocks: {format_input_config_summary()}", flush=True)

    metrics = run_baseline(
        base_path=ADTRANS_ROOT,
        seed=seed,
        n_trials=OPTIMIZER_CONFIG["n_trials"],
    )

    print("\nDone. Metrics (voltage, unscaled):", flush=True)
    for split in ("Train", "Validation", "Test"):
        m = metrics[split]
        print(
            f"  {split:12s} MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R²={m['R2']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
