"""Read train/val/test from source_data/split CSV files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import pandas as pd
from pymatgen.core import Composition

logger = logging.getLogger(__name__)

DATA_PROCESSING_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT_DIR = DATA_PROCESSING_ROOT / "source_data" / "split"
DEFAULT_TRAIN_CSV = DEFAULT_SPLIT_DIR / "train.csv"
DEFAULT_VAL_CSV = DEFAULT_SPLIT_DIR / "val.csv"
DEFAULT_TEST_CSV = DEFAULT_SPLIT_DIR / "test.csv"
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


def parse_formula(formula: str) -> Tuple[List[str], List[float]]:
    comp = Composition(formula)
    elements = [str(e) for e in comp.elements]
    counts = [comp[e] for e in comp.elements]
    total = sum(counts)
    return elements, [c / total for c in counts]


def resolve_split_dir(split_dir: Optional[str | Path] = None) -> Path:
    if split_dir is None:
        return DEFAULT_SPLIT_DIR
    path = Path(split_dir)
    if not path.is_absolute():
        path = DATA_PROCESSING_ROOT / path
    if path.is_file():
        path = path.parent
    return path


def split_csv_path(split_dir: Path, split_name: str) -> Path:
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {split_name}")
    return split_dir / f"{split_name}.csv"


def _battery_ids_from_split_csv(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"split CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, usecols=["battery_id"])
    return df["battery_id"].astype(str).tolist()


def load_split_assignment(split_dir: Optional[str | Path] = None) -> SplitAssignment:
    root = resolve_split_dir(split_dir)
    assignment = SplitAssignment(
        train_battery_ids=_battery_ids_from_split_csv(split_csv_path(root, "train")),
        val_battery_ids=_battery_ids_from_split_csv(split_csv_path(root, "val")),
        test_battery_ids=_battery_ids_from_split_csv(split_csv_path(root, "test")),
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


def prepare_electrode_dataframe(electrode_data_path: str | Path) -> pd.DataFrame:
    path = Path(electrode_data_path)
    if not path.is_absolute():
        path = DATA_PROCESSING_ROOT / path

    if path.is_dir():
        parts = [pd.read_csv(split_csv_path(path, name)) for name in ("train", "val", "test")]
        electrode_df = pd.concat(parts, ignore_index=True)
    else:
        electrode_df = pd.read_csv(path)

    electrode_df["n_elements"] = electrode_df["formula_discharge"].apply(
        lambda x: len(parse_formula(x)[0])
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


def load_element_descriptors(descriptor_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(descriptor_path)
    if "element" in df.columns:
        element_col = "element"
    elif "Symbol" in df.columns:
        element_col = "Symbol"
    else:
        raise ValueError(f"cannot find element column in {descriptor_path}")
    descriptor_cols = [col for col in df.columns if col != element_col]
    return df.set_index(element_col)[descriptor_cols]


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
