from __future__ import annotations

from typing import Dict, Iterable, Literal, Set, Tuple

import pandas as pd

from .split import prepare_electrode_dataframe, structure_series_by_battery_id

CHARGE_STRUCTURE_COLUMN = "charge_structure"
DISCHARGE_STRUCTURE_COLUMN = "discharge_structure"
StructureState = Literal["charge", "discharge"]


def charge_structure_by_battery_id(structure_data_path: str) -> pd.Series:
    return structure_series_by_battery_id(structure_data_path, CHARGE_STRUCTURE_COLUMN)


def discharge_structure_by_battery_id(structure_data_path: str) -> pd.Series:
    return structure_series_by_battery_id(structure_data_path, DISCHARGE_STRUCTURE_COLUMN)


def load_charge_discharge_structure_series(
    structure_data_path: str,
) -> Tuple[pd.Series, pd.Series]:
    return (
        charge_structure_by_battery_id(structure_data_path),
        discharge_structure_by_battery_id(structure_data_path),
    )


def load_voltage_by_battery_id(electrode_data_path: str) -> Dict[str, float]:
    electrode_df = prepare_electrode_dataframe(electrode_data_path)
    return {
        str(row["battery_id"]): float(row["average_voltage"])
        for _, row in electrode_df.iterrows()
    }


def filter_battery_ids_with_charge_discharge(
    battery_ids: Iterable[str],
    charge_index: pd.Series,
    discharge_index: pd.Series,
) -> Set[str]:
    charge_ids = set(charge_index.index.astype(str))
    discharge_ids = set(discharge_index.index.astype(str))
    return {
        str(bid)
        for bid in battery_ids
        if str(bid) in charge_ids and str(bid) in discharge_ids
    }
