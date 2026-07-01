"""Descriptor generation pipeline."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from chgnet.model.model import CHGNet
from pymatgen.core import Composition, Element, Structure
from dataset_split import (
    SplitAssignment,
    load_element_descriptors,
    load_split_assignment,
    parse_formula,
    prepare_electrode_dataframe,
    save_split_csvs_with_assignment,
    structure_series_by_battery_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

Z_ENC_DIM = 20
DEFAULT_DESCRIPTOR_ROOT = Path("descriptor")
DEFAULT_ELECTRODE_CSV = "source_data/split"
DEFAULT_CHGNET_COLUMN_PREFIX = "chg_struct_emb"
DEFAULT_CHGNET_MODEL_NAME = "0.2.0"
DEFAULT_STRUCTURE_DESCRIPTOR_NAME = "chgnet"
CHGNET_CRYSTAL_FEA_DIM = 64

ELECTRODE_ENCODING_DIM = 16
COMPRESSION_STEP_RATIO = 0.5
DEFAULT_ELECTRODE_BASE_SCALE = 1.0

GENERATION_CONFIG = {
    "element_base_types":("skipatom","mat2vec", "magpie"),
    "generate_property_base": True,
    "property_electrode_features": ("band_gap", "energy_above_hull"),
    "generate_structure_base": True,
    "structure_include_charge": True,
    "chgnet_device": "cpu",
}
_ELEMENT_SOURCE_FILES = {
    "magpie": "source_data/magpie.csv",
    "mat2vec": "source_data/mat2vec.csv",
    "skipatom": "source_data/skipatom.csv",
}

_chgnet_model_singleton: Optional[CHGNet] = None
_chgnet_model_sig: Optional[Tuple[str, str]] = None


@dataclass(frozen=True)
class DescriptorSettings:
    max_elements: int
    descriptor_root: Path
    metadata_columns: Tuple[str, ...]
    structure_descriptor_name: str


_settings: Optional[DescriptorSettings] = None


def apply_descriptor_settings(settings: DescriptorSettings) -> None:
    global _settings
    _settings = settings


def _cfg() -> DescriptorSettings:
    if _settings is None:
        apply_descriptor_settings(
            DescriptorSettings(
                max_elements=6,
                descriptor_root=DEFAULT_DESCRIPTOR_ROOT,
                metadata_columns=("average_voltage", "formula_discharge", "battery_id"),
                structure_descriptor_name=DEFAULT_STRUCTURE_DESCRIPTOR_NAME,
            )
        )
    return _settings


def _electrode_compressed_scales(base_scale: float, n: int = 15) -> np.ndarray:
    r = COMPRESSION_STEP_RATIO
    return base_scale * (r ** np.arange(n, dtype=np.float64))


def encode_electrode_scalar_16d(x: float, base_scale: float = DEFAULT_ELECTRODE_BASE_SCALE) -> np.ndarray:
    if not np.isfinite(x):
        raise ValueError(f"electrode scalar must be finite, got {x!r}")
    out = np.zeros(ELECTRODE_ENCODING_DIM, dtype=np.float64)
    out[0] = float(x)
    out[1:] = np.arcsinh(x / _electrode_compressed_scales(base_scale))
    return out


def _clean_electrode_feature_names(electrode_features: Optional[List[str]]) -> List[str]:
    return [name for name in (electrode_features or []) if name]


def _is_electrode_value_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _parse_electrode_bool(value: Any) -> Optional[bool]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0) and np.isfinite(value):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _write_electrode_feature_vector(features: Dict[str, float], feature_name: str, vec: np.ndarray) -> None:
    for i in range(ELECTRODE_ENCODING_DIM):
        features[f"{feature_name}_{i + 1}"] = float(vec[i])


def _assign_electrode_feature_literal(features: Dict[str, float], feature_name: str, literal: float) -> None:
    vec = np.full(ELECTRODE_ENCODING_DIM, literal, dtype=np.float64)
    _write_electrode_feature_vector(features, feature_name, vec)


def _assign_electrode_feature_columns(
    features: Dict[str, float],
    feature_name: str,
    feature_value: Any,
) -> None:
    if _is_electrode_value_missing(feature_value):
        raise ValueError(f"electrode feature {feature_name!r} is missing")
    parsed_bool = _parse_electrode_bool(feature_value)
    if parsed_bool is not None:
        _assign_electrode_feature_literal(features, feature_name, 1.0 if parsed_bool else 0.0)
        return
    try:
        numeric_value = float(feature_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"electrode feature {feature_name!r} is not numeric: {feature_value!r}"
        ) from exc
    if not np.isfinite(numeric_value):
        raise ValueError(
            f"electrode feature {feature_name!r} is not finite: {feature_value!r}"
        )
    _write_electrode_feature_vector(
        features, feature_name, encode_electrode_scalar_16d(numeric_value, DEFAULT_ELECTRODE_BASE_SCALE)
    )


def _cuda_is_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        probe = torch.zeros(1, device="cuda")
        probe.add_(1)
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def _resolve_chgnet_device(explicit: Optional[str]) -> str:
    if explicit is not None and str(explicit).strip():
        requested = str(explicit).strip().lower()
    else:
        env = os.environ.get("CHGNET_DEVICE", "").strip()
        requested = env if env else "auto"
    if requested in {"auto", "cuda"} and _cuda_is_usable():
        return "cuda"
    return "cpu"


def _get_chgnet_model(
    use_device: Optional[str] = "cpu",
    model_name: str = DEFAULT_CHGNET_MODEL_NAME,
) -> CHGNet:
    global _chgnet_model_singleton, _chgnet_model_sig
    dev_key = _resolve_chgnet_device(use_device if use_device is not None else "auto")
    sig = (dev_key, str(model_name))
    if _chgnet_model_singleton is not None and _chgnet_model_sig == sig:
        return _chgnet_model_singleton
    _chgnet_model_singleton = CHGNet.load(model_name=model_name, use_device=dev_key)
    _chgnet_model_singleton = _chgnet_model_singleton.to(dev_key)
    _chgnet_model_sig = sig
    return _chgnet_model_singleton


def _structure_to_embedding(structure_json: str, model: CHGNet) -> np.ndarray:
    if not isinstance(structure_json, str):
        structure_json = str(structure_json)
    structure_json = structure_json.strip()
    if not structure_json:
        raise ValueError("empty structure JSON")
    structure = Structure.from_dict(json.loads(structure_json))
    model.eval()
    with torch.no_grad():
        out = model.predict_structure(structure, task="e", return_crystal_feas=True)
    if "crystal_fea" not in out:
        raise KeyError("CHGNet output missing 'crystal_fea'")
    cf = out["crystal_fea"]
    if isinstance(cf, (list, tuple)) and len(cf) == 1:
        cf = cf[0]
    return np.asarray(cf, dtype=np.float64).reshape(-1).astype(np.float32)


def _structure_cell_to_chgnet_descriptor_columns(
    structure_cell: Any,
    model: CHGNet,
    *,
    column_prefix: str = DEFAULT_CHGNET_COLUMN_PREFIX,
    column_offset: int = 0,
    structure_label: str = "structure",
) -> Dict[str, float]:
    if structure_cell is None or (
        isinstance(structure_cell, float) and np.isnan(structure_cell)
    ):
        raise ValueError(f"{structure_label} is empty")
    vec = _structure_to_embedding(structure_cell, model)
    if int(vec.shape[0]) != CHGNET_CRYSTAL_FEA_DIM:
        raise ValueError(
            f"expected {CHGNET_CRYSTAL_FEA_DIM}-dim crystal_fea, got {vec.shape[0]}"
        )
    return {
        f"{column_prefix}_{column_offset + i}": float(vec[i])
        for i in range(CHGNET_CRYSTAL_FEA_DIM)
    }


def _charge_discharge_structure_to_chgnet_descriptor_columns(
    discharge_structure_cell: Any,
    charge_structure_cell: Any,
    model: CHGNet,
    *,
    column_prefix: str = DEFAULT_CHGNET_COLUMN_PREFIX,
    battery_id: Optional[str] = None,
) -> Dict[str, float]:
    discharge_tokens = _structure_cell_to_chgnet_descriptor_columns(
        discharge_structure_cell,
        model,
        column_prefix=column_prefix,
        column_offset=CHGNET_CRYSTAL_FEA_DIM,
        structure_label="discharge_structure",
    )
    charge_tokens = _structure_cell_to_chgnet_descriptor_columns(
        charge_structure_cell,
        model,
        column_prefix=column_prefix,
        column_offset=0,
        structure_label="charge_structure",
    )
    return {**charge_tokens, **discharge_tokens}


def extract_delithiation_element(battery_id: str) -> str:
    text = str(battery_id).strip()
    if "_" not in text:
        raise ValueError(f"battery_id has no element suffix: {battery_id!r}")
    return text.rsplit("_", 1)[-1]


def order_elements_delithiation_first(
    formula: str,
    battery_id: str,
) -> List[Tuple[str, float]]:
    elements, ratios = parse_formula(formula)
    elem_to_ratio = dict(zip(elements, ratios))
    delithiation = extract_delithiation_element(battery_id)

    ordered: List[Tuple[str, float]] = []
    if delithiation in elem_to_ratio:
        ordered.append((delithiation, float(elem_to_ratio[delithiation])))
    else:
        ordered.append((delithiation, 1.0))

    others = sorted(
        (e for e in elements if e != delithiation),
        key=lambda sym: Element(sym).Z,
    )
    for element in others:
        ordered.append((element, float(elem_to_ratio[element])))
    return ordered


def safe_divide(a, b, default=0.0):
    return a / b if b != 0 else default


def weighted_quantile(values, weights, quantiles):
    if len(values) == 0:
        return np.nan
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumsum = np.cumsum(sorted_weights)
    cumsum = cumsum / cumsum[-1]
    return np.interp(quantiles, cumsum, sorted_values)


def robust_weighted_mean(values, weights):
    if len(values) == 0:
        return np.nan
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad > 0:
        threshold = 3.5 * mad
        mask = np.abs(values - median) <= threshold
        if np.sum(mask) > 0:
            values = values[mask]
            weights = weights[mask]
    return np.average(values, weights=weights)


def safe_log(x, default=0.0):
    return np.log(x) if x > 0 else default


def normalize_weights(weights):
    if len(weights) == 0:
        return weights
    return weights / np.sum(weights)


def compound_to_features(
    formula: str,
    element_df: pd.DataFrame,
    electrode_features: Optional[List[str]] = None,
    electrode_data: Optional[pd.DataFrame] = None,
) -> pd.Series:
    comp = Composition(formula)
    elements = [str(e) for e in comp.elements]
    counts = [comp[e] for e in comp.elements]
    total = sum(counts)
    ratios = [c / total for c in counts]
    features: Dict[str, float] = {}

    missing_elements = [el for el in elements if el not in element_df.index]
    if missing_elements:
        raise KeyError(
            f"formula {formula!r} contains elements not in element table: {missing_elements}"
        )

    for prop in element_df.columns:
        values = []
        weights = []
        for el, ratio in zip(elements, ratios):
            if el in element_df.index and not pd.isna(element_df.loc[el, prop]):
                values.append(element_df.loc[el, prop])
                weights.append(ratio)
        if not values:
            continue
        arr = np.array(values)
        wts = normalize_weights(np.array(weights))
        features[f"{prop}_mean"] = robust_weighted_mean(arr, wts)
        features[f"{prop}_max"] = np.max(arr)
        features[f"{prop}_min"] = np.min(arr)
        features[f"{prop}_range"] = np.max(arr) - np.min(arr)
        mean = features[f"{prop}_mean"]
        features[f"{prop}_std"] = np.sqrt(np.average((arr - mean) ** 2, weights=wts))
        features[f"{prop}_q25"] = weighted_quantile(arr, wts, 0.25)
        features[f"{prop}_q75"] = weighted_quantile(arr, wts, 0.75)
        features[f"{prop}_iqr"] = features[f"{prop}_q75"] - features[f"{prop}_q25"]
        if len(arr) > 1:
            diffs = np.diff(arr)
            features[f"{prop}_max_diff"] = np.max(np.abs(diffs))
            features[f"{prop}_mean_diff"] = np.mean(np.abs(diffs))
        features[f"{prop}_weighted_median"] = weighted_quantile(arr, wts, 0.5)
        if len(arr) > 1:
            features[f"{prop}_entropy"] = -np.sum(wts * np.array([safe_log(w) for w in wts]))
        features[f"{prop}_max_ratio"] = np.max(wts)
        features[f"{prop}_max_min_ratio"] = safe_divide(np.max(arr), np.min(arr))
        features[f"{prop}_max_mean_ratio"] = safe_divide(np.max(arr), features[f"{prop}_mean"])
        sorted_indices = np.argsort(arr)
        features[f"{prop}_sorted_ratio"] = safe_divide(
            wts[sorted_indices[-1]], wts[sorted_indices[0]]
        )

    active_electrode_features = _clean_electrode_feature_names(electrode_features)
    if active_electrode_features:
        if electrode_data is None:
            raise ValueError("electrode_data is required when electrode_features is set")
        electrode_row = electrode_data[electrode_data["formula_discharge"] == formula]
        if electrode_row.empty:
            raise KeyError(f"formula {formula!r} not found in electrode data")
        for feature_name in active_electrode_features:
            if feature_name not in electrode_row.columns:
                raise KeyError(f"electrode feature column {feature_name!r} not in electrode data")
            _assign_electrode_feature_columns(
                features,
                feature_name,
                electrode_row[feature_name].iloc[0],
            )

    if len(features) == 0:
        raise ValueError(f"no features generated for formula {formula!r}")
    return pd.Series(features)


def _build_property_merged_df(
    electrode_df: pd.DataFrame,
    desc_name: str,
    desc_data: pd.DataFrame,
    electrode_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    max_elements = _cfg().max_elements
    property_dfs_by_n: Dict[int, pd.DataFrame] = {}

    for n_elements in sorted(electrode_df["n_elements"].unique()):
        if n_elements < 2 or n_elements > max_elements:
            continue
        n_elements_df = electrode_df[electrode_df["n_elements"] == n_elements]
        material_descriptors = []
        voltages = []
        formulas = []
        battery_ids = []

        for _, row in n_elements_df.iterrows():
            formula = row["formula_discharge"]
            descriptor = compound_to_features(
                formula,
                desc_data,
                electrode_features,
                electrode_df,
            )
            material_descriptors.append(descriptor)
            voltages.append(row["average_voltage"])
            formulas.append(formula)
            battery_ids.append(row["battery_id"])

        if not material_descriptors:
            raise ValueError(
                f"cannot generate {desc_name} property descriptor for {n_elements} elements"
            )

        output_df = pd.DataFrame(material_descriptors)
        output_df["average_voltage"] = voltages
        output_df["formula_discharge"] = formulas
        output_df["battery_id"] = battery_ids
        property_dfs_by_n[n_elements] = output_df

    if not property_dfs_by_n:
        raise ValueError(f"cannot generate any {desc_name} property descriptors")

    reference_n = max_elements if max_elements in property_dfs_by_n else max(property_dfs_by_n)
    max_columns = property_dfs_by_n[reference_n].columns.tolist()
    merged_parts = [property_dfs_by_n[reference_n]]
    for n_elements in sorted(property_dfs_by_n):
        if n_elements == reference_n:
            continue
        df = property_dfs_by_n[n_elements].copy()
        for col in max_columns:
            if col not in df.columns:
                df[col] = 0
        merged_parts.append(df[max_columns])
    return pd.concat(merged_parts, ignore_index=True)


def generate_property_base_descriptors(
    electrode_data_path: str,
    split_assignment: SplitAssignment,
    electrode_features: Optional[Sequence[str]] = None,
) -> None:
    cfg = _cfg()
    features = _clean_electrode_feature_names(list(electrode_features or ()))
    if features:
        logger.info(
            "\n--- generate property_base descriptors (magpie stats + asinh electrode 16d): %s ---",
            features,
        )
    else:
        logger.info("\n--- generate property_base descriptors (magpie element statistics only) ---")

    electrode_df = prepare_electrode_dataframe(electrode_data_path)
    electrode_df = electrode_df[
        (electrode_df["n_elements"] >= 2) & (electrode_df["n_elements"] <= cfg.max_elements)
    ]
    desc_data = load_element_descriptors("source_data/magpie.csv")
    merged_df = _build_property_merged_df(
        electrode_df,
        "magpie",
        desc_data,
        features,
    )
    output_dir = cfg.descriptor_root / "property_base" / "magpie"
    save_split_csvs_with_assignment(
        merged_df, output_dir, split_assignment, metadata_columns=_cfg().metadata_columns
    )
    logger.info("saved property descriptors to %s", output_dir)
    logger.info("--- generate property_base descriptors completed ---")


def _build_structure_merged_df(    electrode_data_path: str,
    structure_data_path: str,
    scope_battery_ids: Set[str],
    chgnet_device: Optional[str] = "cpu",
    chgnet_model_name: str = DEFAULT_CHGNET_MODEL_NAME,
    structure_column_prefix: str = DEFAULT_CHGNET_COLUMN_PREFIX,
    *,
    include_charge: bool = False,
) -> pd.DataFrame:
    cfg = _cfg()
    electrode_df = prepare_electrode_dataframe(electrode_data_path)
    electrode_df = electrode_df[
        (electrode_df["n_elements"] >= 2) & (electrode_df["n_elements"] <= cfg.max_elements)
    ]
    electrode_df = electrode_df[electrode_df["battery_id"].astype(str).isin(scope_battery_ids)]

    discharge_structure_index = structure_series_by_battery_id(
        structure_data_path, "discharge_structure"
    )
    charge_structure_index = (
        structure_series_by_battery_id(structure_data_path, "charge_structure")
        if include_charge
        else None
    )
    chgnet_model = _get_chgnet_model(use_device=chgnet_device, model_name=chgnet_model_name)
    logger.info(
        "loaded discharge_structure rows=%d%s",
        len(discharge_structure_index),
        f", charge_structure rows={len(charge_structure_index)}" if include_charge else "",
    )

    structure_rows = []
    structure_cache: Dict[str, Dict[str, float]] = {}

    for _, row in electrode_df.iterrows():
        formula = row["formula_discharge"]
        battery_id = str(row["battery_id"])
        if battery_id not in discharge_structure_index.index:
            raise KeyError(
                f"battery_id {battery_id!r} not found in discharge_structure for formula {formula!r}"
            )
        if include_charge and (
            charge_structure_index is None or battery_id not in charge_structure_index.index
        ):
            raise KeyError(
                f"battery_id {battery_id!r} not found in charge_structure for formula {formula!r}"
            )
        if battery_id not in structure_cache:
            if include_charge:
                structure_cache[battery_id] = _charge_discharge_structure_to_chgnet_descriptor_columns(
                    discharge_structure_index.loc[battery_id],
                    charge_structure_index.loc[battery_id],
                    chgnet_model,
                    column_prefix=structure_column_prefix,
                    battery_id=battery_id,
                )
            else:
                structure_cache[battery_id] = _structure_cell_to_chgnet_descriptor_columns(
                    discharge_structure_index.loc[battery_id],
                    chgnet_model,
                    column_prefix=structure_column_prefix,
                    column_offset=0,
                    structure_label="discharge_structure",
                )
        record = dict(structure_cache[battery_id])
        record["average_voltage"] = row["average_voltage"]
        record["formula_discharge"] = formula
        record["battery_id"] = row["battery_id"]
        structure_rows.append(record)

    if not structure_rows:
        raise ValueError("no structure descriptor rows generated")
    return pd.DataFrame(structure_rows)


def generate_structure_base_descriptors(
    electrode_data_path: str,
    structure_data_path: str,
    split_assignment: SplitAssignment,
    *,
    chgnet_device: Optional[str] = "cpu",
    chgnet_model_name: str = DEFAULT_CHGNET_MODEL_NAME,
    structure_column_prefix: str = DEFAULT_CHGNET_COLUMN_PREFIX,
    structure_descriptor_name: Optional[str] = None,
    include_charge: bool = False,
) -> None:
    cfg = _cfg()
    descriptor_name = structure_descriptor_name or cfg.structure_descriptor_name
    out_dim = CHGNET_CRYSTAL_FEA_DIM * (2 if include_charge else 1)
    logger.info(
        "\n--- generate structure_base descriptors (%s, pretrained CHGNet %s, %dd) ---",
        descriptor_name,
        chgnet_model_name,
        out_dim,
    )

    structure_merged_df = _build_structure_merged_df(
        electrode_data_path,
        structure_data_path,
        scope_battery_ids=split_assignment.all_battery_ids(),
        chgnet_device=chgnet_device,
        chgnet_model_name=chgnet_model_name,
        structure_column_prefix=structure_column_prefix,
        include_charge=include_charge,
    )
    if structure_merged_df.empty:
        raise ValueError("structure descriptor dataframe is empty")

    output_dir = cfg.descriptor_root / "structure_base" / descriptor_name
    save_split_csvs_with_assignment(
        structure_merged_df, output_dir, split_assignment, metadata_columns=_cfg().metadata_columns
    )
    logger.info("saved structure descriptors to %s", output_dir)
    logger.info("--- generate structure_base descriptors completed ---")


def run_descriptor_generation(
    electrode_data_path: str = DEFAULT_ELECTRODE_CSV,
    generation_config: Optional[Dict[str, Any]] = None,
) -> None:
    cfg_map = {**GENERATION_CONFIG, **(generation_config or {})}
    tasks = []
    if cfg_map.get("element_base_types"):
        tasks.append(f"element_base={cfg_map['element_base_types']}")
    if cfg_map.get("generate_property_base", False):
        feat = cfg_map.get("property_electrode_features") or ()
        if feat:
            tasks.append(f"property_base(magpie+electrode={feat})")
        else:
            tasks.append("property_base(magpie-only)")
    if cfg_map.get("generate_structure_base"):
        if cfg_map.get("structure_include_charge"):
            tasks.append("structure_base(charge+discharge 128d)")
        else:
            tasks.append("structure_base(discharge 64d)")
    if not tasks:
        raise ValueError(
            "Nothing to generate: enable element_base_types, generate_property_base, "
            "or generate_structure_base in GENERATION_CONFIG."
        )
    logger.info("planned generation: %s", "; ".join(tasks))

    apply_descriptor_settings(
        DescriptorSettings(
            max_elements=6,
            descriptor_root=DEFAULT_DESCRIPTOR_ROOT,
            metadata_columns=("average_voltage", "formula_discharge", "battery_id"),
            structure_descriptor_name=DEFAULT_STRUCTURE_DESCRIPTOR_NAME,
        )
    )

    split_assignment = load_split_assignment()

    if cfg_map.get("element_base_types"):
        generate_element_base_descriptors_ordered(
            electrode_data_path,
            split_assignment=split_assignment,
            element_types=cfg_map["element_base_types"],
        )
    if cfg_map.get("generate_property_base", False):
        generate_property_base_descriptors(
            electrode_data_path,
            split_assignment=split_assignment,
            electrode_features=cfg_map.get("property_electrode_features") or (),
        )
    if cfg_map.get("generate_structure_base"):
        generate_structure_base_descriptors(
            electrode_data_path,
            structure_data_path=electrode_data_path,
            split_assignment=split_assignment,
            chgnet_device=cfg_map.get("chgnet_device", "cpu"),
            chgnet_model_name=DEFAULT_CHGNET_MODEL_NAME,
            structure_descriptor_name=DEFAULT_STRUCTURE_DESCRIPTOR_NAME,
            include_charge=bool(cfg_map.get("structure_include_charge", False)),
        )

    _write_meta_report(DEFAULT_DESCRIPTOR_ROOT)
    logger.info("descriptor generation completed")


def encode_atomic_number(z: int, dim: int = Z_ENC_DIM) -> np.ndarray:
    if z <= 0:
        raise ValueError(f"atomic number must be positive, got {z}")
    out = np.zeros(dim, dtype=np.float64)
    for i in range(dim // 2):
        denom = 10000 ** (2 * i / dim)
        out[2 * i] = np.sin(z / denom)
        out[2 * i + 1] = np.cos(z / denom)
    if dim % 2 == 1:
        out[-1] = np.sin(z / (10000 ** ((dim - 1) / dim)))
    return out


def build_slot_column_names(
    element_descriptor_columns: Sequence[str],
    max_slots: int,
    z_enc_dim: int = Z_ENC_DIM,
) -> List[str]:
    names: List[str] = []
    for slot in range(1, max_slots + 1):
        for prop in element_descriptor_columns:
            names.append(f"{prop}_{slot}")
        for j in range(1, z_enc_dim + 1):
            names.append(f"z_enc_{j}_{slot}")
    return names


def features_per_element_slot(element_descriptor_dim: int, z_enc_dim: int = Z_ENC_DIM) -> int:
    return int(element_descriptor_dim) + int(z_enc_dim)


def generate_material_descriptor_ordered(
    formula: str,
    battery_id: str,
    element_descriptors: pd.DataFrame,
    *,
    max_slots: int,
    z_enc_dim: int = Z_ENC_DIM,
) -> Tuple[List[float], List[str]]:
    ordered_elements = order_elements_delithiation_first(formula, battery_id)
    prop_names = list(element_descriptors.columns)
    column_names = build_slot_column_names(prop_names, max_slots, z_enc_dim)

    slot_vectors: List[np.ndarray] = []
    slot_weights: List[float] = []

    for element, ratio in ordered_elements:
        if element not in element_descriptors.index:
            raise KeyError(
                f"element {element!r} not found in descriptors for formula {formula!r}, "
                f"battery_id {battery_id!r}"
            )

        elem_vec = element_descriptors.loc[element].values.astype(np.float64)
        weighted = elem_vec * ratio
        z_enc = encode_atomic_number(int(Element(element).Z), z_enc_dim)
        slot_vectors.append(np.concatenate([weighted, z_enc]))
        slot_weights.append(max(ratio, 0.0))

    weights = np.asarray(slot_weights, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(slot_vectors), dtype=np.float64) / len(slot_vectors)
    else:
        weights = weights / weights.sum()
    pad_vector = np.average(np.stack(slot_vectors, axis=0), axis=0, weights=weights)

    full_slots = slot_vectors + [pad_vector] * max(0, max_slots - len(slot_vectors))
    descriptor = np.concatenate(full_slots[:max_slots], axis=0)
    if np.isnan(descriptor).any():
        raise ValueError(
            f"NaN in ordered descriptor for formula {formula!r}, battery_id {battery_id!r}"
        )
    return descriptor.tolist(), column_names


def _build_element_merged_df_ordered(
    electrode_df: pd.DataFrame,
    desc_name: str,
    desc_data: pd.DataFrame,
) -> pd.DataFrame:
    cfg = _cfg()
    max_elements = cfg.max_elements
    element_dfs_by_n: Dict[int, pd.DataFrame] = {}
    element_dim = int(desc_data.shape[1])
    slot_dim = features_per_element_slot(element_dim)

    for n_elements in range(2, max_elements + 1):
        n_elements_df = electrode_df[electrode_df["n_elements"] == n_elements]
        if n_elements_df.empty:
            continue

        material_descriptors: List[List[float]] = []
        voltages: List[float] = []
        formulas: List[str] = []
        battery_ids: List[str] = []
        column_names: Optional[List[str]] = None

        for _, row in n_elements_df.iterrows():
            formula = row["formula_discharge"]
            battery_id = str(row["battery_id"])
            descriptor, descriptor_names = generate_material_descriptor_ordered(
                formula,
                battery_id,
                desc_data,
                max_slots=max_elements,
            )

            material_descriptors.append(descriptor)
            voltages.append(row["average_voltage"])
            formulas.append(formula)
            battery_ids.append(row["battery_id"])
            if column_names is None:
                column_names = descriptor_names

        if not material_descriptors:
            raise ValueError(
                f"cannot generate {desc_name} ordered descriptor for {n_elements} elements"
            )

        expected_len = max_elements * slot_dim
        if column_names is None or len(column_names) != expected_len:
            raise ValueError(
                f"column layout mismatch for {desc_name}: "
                f"expected {expected_len}, got {0 if column_names is None else len(column_names)}"
            )

        output_df = pd.DataFrame(material_descriptors, columns=column_names)
        output_df["average_voltage"] = voltages
        output_df["formula_discharge"] = formulas
        output_df["battery_id"] = battery_ids
        element_dfs_by_n[n_elements] = output_df

    if max_elements not in element_dfs_by_n:
        raise ValueError(
            f"cannot build max{max_elements} {desc_name} ordered descriptor without "
            f"{max_elements}-element data"
        )

    return pd.concat(list(element_dfs_by_n.values()), ignore_index=True)


def generate_element_base_descriptors_ordered(
    electrode_data_path: str,
    split_assignment: SplitAssignment,
    element_types: Optional[Sequence[str]] = None,
) -> None:
    cfg = _cfg()
    types = tuple(element_types or ())
    if not types:
        logger.info("skip element_base (element_base_types is empty)")
        return

    unknown = [t for t in types if t not in _ELEMENT_SOURCE_FILES]
    if unknown:
        raise ValueError(
            f"Unknown element_base_types: {unknown}. "
            f"Choose from {list(_ELEMENT_SOURCE_FILES)}"
        )

    logger.info("\n--- generate element_base descriptors (ordered + Z encoding): %s ---", types)

    electrode_df = prepare_electrode_dataframe(electrode_data_path)
    electrode_df = electrode_df[
        (electrode_df["n_elements"] >= 2) & (electrode_df["n_elements"] <= cfg.max_elements)
    ]

    for desc_name in types:
        path = _ELEMENT_SOURCE_FILES[desc_name]
        try:
            desc_data = load_element_descriptors(path)
        except Exception as exc:
            raise RuntimeError(f"load {desc_name} descriptor failed: {exc}") from exc

        slot_dim = features_per_element_slot(desc_data.shape[1])
        logger.info(
            "load %s descriptor: element_dim=%d, slot_dim=%d (+%d Z enc)",
            desc_name,
            desc_data.shape[1],
            slot_dim,
            Z_ENC_DIM,
        )

        merged_df = _build_element_merged_df_ordered(electrode_df, desc_name, desc_data)

        output_dir = cfg.descriptor_root / "element_base" / desc_name
        save_split_csvs_with_assignment(
            merged_df, output_dir, split_assignment, metadata_columns=_cfg().metadata_columns
        )
        logger.info("saved ordered %s element descriptors to %s", desc_name, output_dir)

    logger.info("--- generate element_base descriptors (ordered) completed ---")


def _write_meta_report(descriptor_root: Path) -> None:
    report_dir = Path("dataprocess_summary_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    meta_path = report_dir / "descriptor_meta.txt"
    lines = [
        "descriptors_generator.py",
        f"descriptor_root: {descriptor_root}",
        "outputs:",
        "  element_base/{magpie,mat2vec,skipatom}",
        "  property_base/magpie",
        f"  structure_base/{DEFAULT_STRUCTURE_DESCRIPTOR_NAME} (CHGNet {DEFAULT_CHGNET_MODEL_NAME})",
        f"z_encoding_dim: {Z_ENC_DIM}",
    ]
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote metadata to %s", meta_path)


if __name__ == "__main__":
    run_descriptor_generation()
