from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .split import (
    SplitAssignment,
    load_split_assignment,
    prepare_electrode_dataframe,
    save_split_csvs_with_assignment,
    structure_series_by_battery_id,
)

__all__ = [
    "DescriptorSettings",
    "SplitAssignment",
    "apply_descriptor_settings",
    "load_split_assignment",
    "prepare_electrode_dataframe",
    "save_split_csvs_with_assignment",
    "structure_series_by_battery_id",
]


@dataclass(frozen=True)
class DescriptorSettings:
    max_elements: int
    descriptor_root: Path


_settings: DescriptorSettings | None = None


def apply_descriptor_settings(settings: DescriptorSettings) -> None:
    global _settings
    _settings = settings


def _cfg() -> DescriptorSettings:
    if _settings is None:
        apply_descriptor_settings(
            DescriptorSettings(
                max_elements=6,
                descriptor_root=Path("descriptors"),
            )
        )
    return _settings
