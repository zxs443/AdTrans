"""Global split from CSV; inner train/val split within global train."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

DATA_PROCESSING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_COLUMNS = ("average_voltage", "formula_discharge", "battery_id")


@dataclass
class SplitAssignment:
    train_battery_ids: List[str]
    val_battery_ids: List[str]
    test_battery_ids: List[str]

    def _validate_unique(self) -> None:
        all_ids = self.train_battery_ids + self.val_battery_ids + self.test_battery_ids
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("duplicate battery_id across train/val/test")

    def ordered_battery_ids(self, split_name: str) -> List[str]:
        if split_name == "train":
            return self.train_battery_ids
        if split_name == "val":
            return self.val_battery_ids
        if split_name == "test":
            return self.test_battery_ids
        raise ValueError(f"unknown split: {split_name}")

    def all_battery_ids(self) -> Set[str]:
        return set(self.train_battery_ids + self.val_battery_ids + self.test_battery_ids)


def _resolve_split_dir(split_dir: str | Path | None) -> Path:
    if split_dir is None:
        return DATA_PROCESSING_ROOT / "source_data" / "split"
    path = Path(split_dir)
    if not path.is_absolute():
        path = DATA_PROCESSING_ROOT / path
    if path.is_file():
        path = path.parent
    return path


def _split_csv_path(split_dir: Path, split_name: str) -> Path:
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {split_name}")
    return split_dir / f"{split_name}.csv"


def _battery_ids_from_split_csv(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"split CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, usecols=["battery_id"])
    return df["battery_id"].astype(str).tolist()


def load_split_assignment(split_dir: str | Path | None = None) -> SplitAssignment:
    root = _resolve_split_dir(split_dir)
    assignment = SplitAssignment(
        train_battery_ids=_battery_ids_from_split_csv(_split_csv_path(root, "train")),
        val_battery_ids=_battery_ids_from_split_csv(_split_csv_path(root, "val")),
        test_battery_ids=_battery_ids_from_split_csv(_split_csv_path(root, "test")),
    )
    assignment._validate_unique()
    logger.info(
        "loaded split from %s: train=%d val=%d test=%d",
        root,
        len(assignment.train_battery_ids),
        len(assignment.val_battery_ids),
        len(assignment.test_battery_ids),
    )
    return assignment


def _parse_formula(formula: str) -> Tuple[List[str], List[float]]:
    comp = Composition(formula)
    elements = [str(e) for e in comp.elements]
    counts = [comp[e] for e in comp.elements]
    total = sum(counts)
    return elements, [c / total for c in counts]


def prepare_electrode_dataframe(electrode_data_path: str | Path) -> pd.DataFrame:
    path = Path(electrode_data_path)
    if not path.is_absolute():
        path = DATA_PROCESSING_ROOT / path

    if path.is_dir():
        parts = [pd.read_csv(_split_csv_path(path, name)) for name in ("train", "val", "test")]
        electrode_df = pd.concat(parts, ignore_index=True)
    else:
        electrode_df = pd.read_csv(path)

    electrode_df["n_elements"] = electrode_df["formula_discharge"].apply(
        lambda x: len(_parse_formula(x)[0])
    )
    return electrode_df


def structure_series_by_battery_id(electrode_data_path: str | Path, column: str) -> pd.Series:
    structure_df = prepare_electrode_dataframe(electrode_data_path)[["battery_id", column]].copy()
    structure_df["battery_id"] = structure_df["battery_id"].astype(str)
    if structure_df["battery_id"].duplicated().any():
        n_dup = int(structure_df["battery_id"].duplicated().sum())
        logger.warning("electrode data: %d duplicate battery_id; keep first", n_dup)
        structure_df = structure_df.loc[~structure_df["battery_id"].duplicated(keep="first")]
    return structure_df.set_index("battery_id", verify_integrity=True)[column]


def order_descriptor_columns(
    df: pd.DataFrame,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> pd.DataFrame:
    metadata_columns = list(metadata_columns)
    feature_cols = [col for col in df.columns if col not in metadata_columns]
    missing = [col for col in metadata_columns if col not in df.columns]
    if missing:
        raise ValueError(f"missing metadata columns: {missing}")
    return df[feature_cols + metadata_columns]


def save_split_csvs_with_assignment(
    df: pd.DataFrame,
    output_dir: Path,
    split_assignment: SplitAssignment,
    *,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
    strict: bool = True,
) -> None:
    df = order_descriptor_columns(df, metadata_columns).copy()
    df["_battery_id_key"] = df["battery_id"].astype(str)
    if df["_battery_id_key"].duplicated().any():
        dupes = df["_battery_id_key"][df["_battery_id_key"].duplicated()].unique().tolist()
        raise ValueError(f"duplicate battery_id: {dupes[:5]}")

    indexed = df.set_index("_battery_id_key", drop=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        ordered_ids = split_assignment.ordered_battery_ids(split_name)
        missing_ids = [bid for bid in ordered_ids if bid not in indexed.index]
        if strict and missing_ids:
            raise ValueError(
                f"{output_dir} [{split_name}]: missing battery_ids: {missing_ids[:5]}"
            )
        rows = [indexed.loc[bid] for bid in ordered_ids if bid in indexed.index]
        out_df = pd.DataFrame(rows).drop(columns=["_battery_id_key"]) if rows else df.iloc[0:0].copy()
        out_path = output_dir / f"{split_name}.csv"
        out_df.to_csv(out_path, index=False, float_format="%.15g")
        skip = f", skipped {len(missing_ids)}" if missing_ids else ""
        logger.info("saved %s split (%d%s) -> %s", split_name, len(rows), skip, out_path)


@dataclass(frozen=True)
class ChgnetFinetuneSplit:
    """Inner train/val partition of global train (val/test unchanged)."""

    chgnet_train_ids: List[str]
    chgnet_inner_val_ids: List[str]
    projv2_train_ids: List[str]
    projv2_val_ids: List[str]
    projv2_test_ids: List[str]

    def to_dict(self) -> dict:
        return {
            "chgnet_train_ids": self.chgnet_train_ids,
            "chgnet_inner_val_ids": self.chgnet_inner_val_ids,
            "projv2_train_ids": self.projv2_train_ids,
            "projv2_val_ids": self.projv2_val_ids,
            "projv2_test_ids": self.projv2_test_ids,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ChgnetFinetuneSplit":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def stratified_battery_split(
    battery_ids: Sequence[str],
    targets: Sequence[float],
    val_ratio: float,
    seed: int,
) -> tuple[List[str], List[str]]:
    ids = np.asarray(list(battery_ids), dtype=object)
    y = np.asarray(list(targets), dtype=np.float64)
    if len(ids) == 0:
        raise ValueError("Cannot split empty battery_id list.")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1).")

    q_bins = min(6, len(np.unique(y)))
    while q_bins >= 2:
        try:
            bins = pd.qcut(y, q=q_bins, labels=False, duplicates="drop")
            break
        except ValueError:
            q_bins -= 1
    else:
        bins = np.zeros(len(y), dtype=int)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), bins))
    train_ids = ids[train_idx].tolist()
    val_ids = ids[val_idx].tolist()
    if not train_ids or not val_ids:
        raise ValueError("CHGNet inner split produced an empty train or validation set.")
    return train_ids, val_ids


def build_chgnet_finetune_split(
    projv2_train_ids: Sequence[str],
    projv2_val_ids: Sequence[str],
    projv2_test_ids: Sequence[str],
    voltage_by_battery_id: Dict[str, float],
    inner_val_ratio: float,
    split_seed: int,
    inner_val_seed_offset: int,
) -> ChgnetFinetuneSplit:
    missing = [bid for bid in projv2_train_ids if bid not in voltage_by_battery_id]
    if missing:
        raise KeyError(
            f"Missing average_voltage for {len(missing)} train battery_id(s), "
            f"examples: {missing[:3]}"
        )

    train_targets = [voltage_by_battery_id[bid] for bid in projv2_train_ids]
    chgnet_train_ids, chgnet_inner_val_ids = stratified_battery_split(
        projv2_train_ids,
        train_targets,
        val_ratio=inner_val_ratio,
        seed=split_seed + inner_val_seed_offset,
    )
    return ChgnetFinetuneSplit(
        chgnet_train_ids=chgnet_train_ids,
        chgnet_inner_val_ids=chgnet_inner_val_ids,
        projv2_train_ids=list(projv2_train_ids),
        projv2_val_ids=list(projv2_val_ids),
        projv2_test_ids=list(projv2_test_ids),
    )
