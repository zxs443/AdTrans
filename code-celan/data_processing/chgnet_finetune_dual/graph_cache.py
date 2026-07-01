"""CrystalGraph cache for charge and discharge structures."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch
from chgnet.graph.crystalgraph import CrystalGraph
from pymatgen.core import Structure

from .chgnet_loader import get_chgnet_graph_converter
from .structures import StructureState, load_charge_discharge_structure_series

logger = logging.getLogger(__name__)


def _safe_graph_stem(battery_id: str, state: StructureState) -> str:
    safe = re.sub(r"[^\w\-.]+", "_", str(battery_id))
    return f"{safe}__{state}"


def graph_path(cache_dir: Path, battery_id: str, state: StructureState) -> Path:
    return cache_dir / "graphs" / f"{_safe_graph_stem(battery_id, state)}.pt"


def load_crystal_graph(path: Path) -> CrystalGraph:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, CrystalGraph):
        return obj
    if isinstance(obj, dict):
        return CrystalGraph.from_dict(obj)
    raise TypeError(f"Unsupported graph checkpoint type at {path}: {type(obj)}")


def structure_json_to_graph(structure_json: str, converter) -> CrystalGraph:
    structure = Structure.from_dict(json.loads(structure_json))
    return converter(structure)


def _write_graph(
    graphs_dir: Path,
    cache_dir: Path,
    battery_id: str,
    state: StructureState,
    structure_cell,
    converter,
) -> str:
    out_path = graph_path(cache_dir, battery_id, state)
    graph = structure_json_to_graph(str(structure_cell), converter)
    graph.graph_id = f"{battery_id}__{state}"
    graph.save(fname=out_path.name, save_dir=str(graphs_dir))
    return str(out_path.relative_to(cache_dir))


def build_dual_graph_cache(
    structure_data_path: str,
    battery_ids: Sequence[str],
    cache_dir: str | Path,
    *,
    chgnet_model_name: str = "0.3.0",
    overwrite: bool = False,
) -> dict:
    cache_dir = Path(cache_dir)
    graphs_dir = cache_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    charge_index, discharge_index = load_charge_discharge_structure_series(structure_data_path)
    converter = get_chgnet_graph_converter(model_name=chgnet_model_name)
    manifest: Dict[str, dict] = {}
    failures: List[dict] = []

    for battery_id in battery_ids:
        bid = str(battery_id)
        entry: dict = {}
        missing = []
        if bid not in charge_index.index:
            missing.append("charge_structure")
        if bid not in discharge_index.index:
            missing.append("discharge_structure")
        if missing:
            failures.append({"battery_id": bid, "status": f"missing {'/'.join(missing)}"})
            continue

        try:
            for state, index in (("charge", charge_index), ("discharge", discharge_index)):
                rel_key = f"{state}_graph_path"
                out_path = graph_path(cache_dir, bid, state)
                if out_path.exists() and not overwrite:
                    entry[rel_key] = str(out_path.relative_to(cache_dir))
                    entry[f"{state}_status"] = "cached"
                    continue
                entry[rel_key] = _write_graph(
                    graphs_dir, cache_dir, bid, state, index.loc[bid], converter
                )
                entry[f"{state}_status"] = "built"
            manifest[bid] = entry
        except Exception as exc:
            failures.append({"battery_id": bid, "status": f"graph build failed: {exc}"})

    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failures:
        fail_path = cache_dir / "graph_cache_failures.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        logger.warning("Graph cache failures: %d (see %s)", len(failures), fail_path)

    logger.info(
        "Dual graph cache ready at %s (%d entries, %d failures)",
        cache_dir,
        len(manifest),
        len(failures),
    )
    return {"manifest": manifest, "failures": failures, "cache_dir": str(cache_dir)}


def load_graph_for_battery_id(
    cache_dir: str | Path,
    battery_id: str,
    *,
    state: StructureState,
) -> Optional[CrystalGraph]:
    cache_dir = Path(cache_dir)
    bid = str(battery_id)
    manifest_path = cache_dir / "manifest.json"
    rel_path: Optional[str] = None

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest.get(bid) or {}
        rel_path = entry.get(f"{state}_graph_path")

    if rel_path is None:
        candidate = graph_path(cache_dir, bid, state)
        if not candidate.exists():
            return None
        rel_path = str(candidate.relative_to(cache_dir))

    return load_crystal_graph(cache_dir / rel_path)


def load_charge_discharge_graphs(
    cache_dir: str | Path,
    battery_id: str,
) -> tuple[Optional[CrystalGraph], Optional[CrystalGraph]]:
    charge = load_graph_for_battery_id(cache_dir, battery_id, state="charge")
    discharge = load_graph_for_battery_id(cache_dir, battery_id, state="discharge")
    return charge, discharge
