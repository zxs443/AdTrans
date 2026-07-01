"""Export finetuned CHGNet structure descriptors."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from chgnet.model.model import CHGNet
from pymatgen.core import Structure

from .chgnet_loader import DEFAULT_CHGNET_COLUMN_PREFIX
from .element_split import _cfg
from .split import (
    SplitAssignment,
    prepare_electrode_dataframe,
    save_split_csvs_with_assignment,
)

from .config import DualStateFinetuneConfig, ExportEmbeddingMode
from .graph_cache import load_charge_discharge_graphs
from .model import DualStateVoltageModel, fuse_crystal_features, load_chgnet_backbone

logger = logging.getLogger(__name__)


def _structure_to_embedding(structure_json: str, model: CHGNet) -> np.ndarray:
    if not isinstance(structure_json, str):
        structure_json = str(structure_json)
    structure_json = structure_json.strip()
    structure = Structure.from_dict(json.loads(structure_json))
    model.eval()
    with torch.no_grad():
        out = model.predict_structure(structure, task="e", return_crystal_feas=True)
    cf = out["crystal_fea"]
    if isinstance(cf, (list, tuple)) and len(cf) == 1:
        cf = cf[0]
    return np.asarray(cf, dtype=np.float64).reshape(-1).astype(np.float32)


def _dual_embedding_from_json(
    charge_json: Any,
    discharge_json: Any,
    model: CHGNet,
    mode: ExportEmbeddingMode,
) -> np.ndarray:
    if mode == "discharge":
        return _structure_to_embedding(discharge_json, model)
    charge_vec = _structure_to_embedding(charge_json, model)
    discharge_vec = _structure_to_embedding(discharge_json, model)
    charge_t = torch.from_numpy(charge_vec)
    discharge_t = torch.from_numpy(discharge_vec)
    fused = fuse_crystal_features(charge_t, discharge_t, mode)
    return fused.numpy().astype(np.float32)


def build_dual_structure_merged_df(
    electrode_data_path: str,
    scope_battery_ids: Set[str],
    *,
    chgnet_checkpoint: Optional[str] = None,
    chgnet_model_name: str = "0.3.0",
    chgnet_device: Optional[str] = "cpu",
    export_embedding_mode: ExportEmbeddingMode = "concat",
    structure_column_prefix: str = DEFAULT_CHGNET_COLUMN_PREFIX,
) -> Optional[pd.DataFrame]:
    cfg = _cfg()
    electrode_df = prepare_electrode_dataframe(electrode_data_path)
    electrode_df = electrode_df[
        (electrode_df["n_elements"] >= 2) & (electrode_df["n_elements"] <= cfg.max_elements)
    ]
    electrode_df = electrode_df[electrode_df["battery_id"].astype(str).isin(scope_battery_ids)]

    chgnet_model = load_chgnet_backbone(
        model_name=chgnet_model_name,
        checkpoint=chgnet_checkpoint,
        device=str(chgnet_device or "cpu"),
    )

    rows = []
    cache: Dict[str, Tuple[Dict[str, float], bool]] = {}
    for _, row in electrode_df.iterrows():
        battery_id = str(row["battery_id"])
        charge_json = row.get("charge_structure")
        discharge_json = row.get("discharge_structure")
        if pd.isna(discharge_json):
            continue
        if export_embedding_mode != "discharge" and pd.isna(charge_json):
            continue
        if battery_id not in cache:
            try:
                vec = _dual_embedding_from_json(
                    charge_json,
                    discharge_json,
                    chgnet_model,
                    export_embedding_mode,
                )
                tokens = {
                    f"{structure_column_prefix}_{i}": float(vec[i])
                    for i in range(int(vec.shape[0]))
                }
                cache[battery_id] = (tokens, True)
            except Exception as exc:
                logger.warning("dual embedding failed for %s: %s", battery_id, str(exc)[:200])
                cache[battery_id] = ({}, False)
        tokens, ok = cache[battery_id]
        if not ok:
            continue
        record = dict(tokens)
        record["average_voltage"] = row["average_voltage"]
        record["formula_discharge"] = row["formula_discharge"]
        record["battery_id"] = row["battery_id"]
        rows.append(record)

    if not rows:
        return None
    return pd.DataFrame(rows)


def build_dual_structure_merged_df_from_graph_cache(
    electrode_data_path: str,
    graph_cache_dir: str,
    scope_battery_ids: Set[str],
    *,
    chgnet_checkpoint: Optional[str] = None,
    chgnet_model_name: str = "0.3.0",
    chgnet_device: Optional[str] = "cpu",
    export_embedding_mode: ExportEmbeddingMode = "concat",
    structure_column_prefix: str = DEFAULT_CHGNET_COLUMN_PREFIX,
) -> Optional[pd.DataFrame]:
    cfg = _cfg()
    electrode_df = prepare_electrode_dataframe(electrode_data_path)
    electrode_df = electrode_df[
        (electrode_df["n_elements"] >= 2) & (electrode_df["n_elements"] <= cfg.max_elements)
    ]
    electrode_df = electrode_df[electrode_df["battery_id"].astype(str).isin(scope_battery_ids)]

    wrapper = DualStateVoltageModel(
        load_chgnet_backbone(
            model_name=chgnet_model_name,
            checkpoint=chgnet_checkpoint,
            device=str(chgnet_device or "cpu"),
        ),
        fusion_mode=export_embedding_mode,
    )
    wrapper.eval()
    device = next(wrapper.parameters()).device

    rows = []
    for _, row in electrode_df.iterrows():
        battery_id = str(row["battery_id"])
        charge_graph, discharge_graph = load_charge_discharge_graphs(graph_cache_dir, battery_id)
        if charge_graph is None or discharge_graph is None:
            continue
        with torch.no_grad():
            if export_embedding_mode == "discharge":
                fused = wrapper.encode_graphs([discharge_graph.to(device)])
            else:
                fused = wrapper.encode_dual(
                    [charge_graph.to(device)], [discharge_graph.to(device)]
                )
        vec = fused.squeeze(0).detach().cpu().numpy().astype(np.float32)
        record = {
            f"{structure_column_prefix}_{i}": float(vec[i]) for i in range(int(vec.shape[0]))
        }
        record["average_voltage"] = row["average_voltage"]
        record["formula_discharge"] = row["formula_discharge"]
        record["battery_id"] = row["battery_id"]
        rows.append(record)

    if not rows:
        return None
    return pd.DataFrame(rows)


def generate_dual_structure_descriptors(
    electrode_data_path: str,
    split_assignment: SplitAssignment,
    *,
    chgnet_checkpoint: Optional[str] = None,
    chgnet_model_name: str = "0.3.0",
    chgnet_device: Optional[str] = "cpu",
    export_embedding_mode: ExportEmbeddingMode = "concat",
    structure_column_prefix: str = DEFAULT_CHGNET_COLUMN_PREFIX,
    structure_descriptor_name: Optional[str] = None,
    descriptor_root: Optional[Path] = None,
    graph_cache_dir: Optional[str] = None,
) -> None:
    cfg = _cfg()
    descriptor_name = structure_descriptor_name or "chgnet_ft_v_dual"
    root = descriptor_root or cfg.descriptor_root
    logger.info(
        "Generating dual structure descriptors (%s, mode=%s)",
        descriptor_name,
        export_embedding_mode,
    )

    if graph_cache_dir:
        merged_df = build_dual_structure_merged_df_from_graph_cache(
            electrode_data_path,
            graph_cache_dir,
            scope_battery_ids=split_assignment.all_battery_ids(),
            chgnet_checkpoint=chgnet_checkpoint,
            chgnet_model_name=chgnet_model_name,
            chgnet_device=chgnet_device,
            export_embedding_mode=export_embedding_mode,
            structure_column_prefix=structure_column_prefix,
        )
    else:
        merged_df = build_dual_structure_merged_df(
            electrode_data_path,
            scope_battery_ids=split_assignment.all_battery_ids(),
            chgnet_checkpoint=chgnet_checkpoint,
            chgnet_model_name=chgnet_model_name,
            chgnet_device=chgnet_device,
            export_embedding_mode=export_embedding_mode,
            structure_column_prefix=structure_column_prefix,
        )

    if merged_df is None or merged_df.empty:
        logger.warning("cannot generate any dual structure descriptor data")
        return

    output_dir = root / "structure_base" / descriptor_name
    save_split_csvs_with_assignment(merged_df, output_dir, split_assignment)
    logger.info("saved dual structure descriptors to %s (dim=%d)", output_dir, merged_df.shape[1] - 3)
