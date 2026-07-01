from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import torch
from chgnet.graph.crystalgraph import CrystalGraph
from torch.utils.data import Dataset

from .graph_cache import load_charge_discharge_graphs

logger = logging.getLogger(__name__)


class DualStateVoltageGraphDataset(Dataset):
    def __init__(
        self,
        battery_ids: Sequence[str],
        voltage_by_battery_id: dict,
        graph_cache_dir: str,
        *,
        preload_graphs: bool = True,
        preload_device: torch.device | str = "cpu",
    ):
        self.graph_cache_dir = graph_cache_dir
        self.preload_graphs = preload_graphs
        self.preload_device = torch.device(preload_device)
        self.samples: List[Tuple[str, float]] = []
        self._charge_graphs: List[CrystalGraph] = []
        self._discharge_graphs: List[CrystalGraph] = []
        missing: List[str] = []

        for bid in battery_ids:
            bid = str(bid)
            if bid not in voltage_by_battery_id:
                raise KeyError(f"Missing average_voltage for battery_id={bid}")
            charge_graph, discharge_graph = load_charge_discharge_graphs(graph_cache_dir, bid)
            if charge_graph is None or discharge_graph is None:
                missing.append(bid)
                continue
            self.samples.append((bid, float(voltage_by_battery_id[bid])))
            if preload_graphs:
                self._charge_graphs.append(charge_graph)
                self._discharge_graphs.append(discharge_graph)

        if missing:
            logger.warning(
                "Skipping %d/%d battery_id(s) without dual graph cache: %s",
                len(missing),
                len(battery_ids),
                missing[:5],
            )
        if not self.samples:
            raise ValueError(
                "DualStateVoltageGraphDataset has zero samples "
                f"({len(missing)} skipped for missing graph cache)."
            )

        if preload_graphs and self.preload_device.type == "cuda":
            logger.info(
                "Preloading %d graph pairs to %s ...",
                len(self._charge_graphs),
                self.preload_device,
            )
            self._charge_graphs = [g.to(self.preload_device) for g in self._charge_graphs]
            self._discharge_graphs = [g.to(self.preload_device) for g in self._discharge_graphs]
            torch.cuda.synchronize()
            logger.info("GPU graph preload complete (%d pairs on %s)", len(self.samples), self.preload_device)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[CrystalGraph, CrystalGraph, torch.Tensor, str]:
        battery_id, voltage = self.samples[idx]
        if self.preload_graphs:
            charge_graph = self._charge_graphs[idx]
            discharge_graph = self._discharge_graphs[idx]
        else:
            charge_graph, discharge_graph = load_charge_discharge_graphs(
                self.graph_cache_dir, battery_id
            )
            if charge_graph is None or discharge_graph is None:
                raise FileNotFoundError(f"Graph missing at runtime for battery_id={battery_id}")
            if self.preload_device.type == "cuda":
                charge_graph = charge_graph.to(self.preload_device)
                discharge_graph = discharge_graph.to(self.preload_device)
        return (
            charge_graph,
            discharge_graph,
            torch.tensor(voltage, dtype=torch.float32),
            battery_id,
        )


def collate_dual_voltage_graphs(batch):
    charge_graphs, discharge_graphs, voltages, battery_ids = zip(*batch)
    return (
        list(charge_graphs),
        list(discharge_graphs),
        torch.stack(voltages, dim=0),
        list(battery_ids),
    )
