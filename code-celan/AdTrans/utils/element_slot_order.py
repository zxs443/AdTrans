"""Encoder element-slot ordering: delithiation ion first, then ascending Z."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from pymatgen.core import Element

from utils.formula_utils import _parse_formula_composition


def extract_delithiation_element(battery_id: str) -> str:
    text = str(battery_id).strip()
    if "_" not in text:
        raise ValueError(f"battery_id has no element suffix: {battery_id!r}")
    return text.rsplit("_", 1)[-1]


def order_elements_delithiation_first(
    formula: str,
    battery_id: str,
) -> List[Tuple[str, float]]:
    """Return (element, stoichiometric ratio) in encoder slot order."""
    comp = _parse_formula_composition(formula)
    if comp is None:
        raise ValueError(f"cannot parse formula: {formula!r}")

    elem_to_ratio = {str(el): float(comp[el]) for el in comp.elements}
    total = sum(elem_to_ratio.values())
    if total > 0:
        elem_to_ratio = {k: v / total for k, v in elem_to_ratio.items()}

    delithiation = extract_delithiation_element(battery_id)
    ordered: List[Tuple[str, float]] = []
    if delithiation in elem_to_ratio:
        ordered.append((delithiation, elem_to_ratio[delithiation]))
    else:
        ordered.append((delithiation, 1.0))

    others = sorted(
        (e for e in elem_to_ratio if e != delithiation),
        key=lambda sym: Element(sym).Z,
    )
    for element in others:
        ordered.append((element, elem_to_ratio[element]))
    return ordered


def encoder_slot_element_symbols(
    formula: str,
    battery_id: str,
    max_slots: int = 6,
) -> List[str]:
    """Element symbols for encoder slots 0..n-1 (padding slots omitted)."""
    return [
        element
        for element, _ in order_elements_delithiation_first(formula, battery_id)
    ][:max_slots]


def build_slot_label_arrays(
    formulas: Sequence[str],
    battery_ids: Sequence[str],
    n_slots: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """slot_element_z [N, n_slots], slot_count [N, n_slots] in encoder slot order."""
    n = len(formulas)
    if len(battery_ids) != n:
        raise ValueError(
            f"formulas length ({n}) != battery_ids length ({len(battery_ids)})"
        )

    slot_element_z = np.zeros((n, n_slots), dtype=np.int64)
    slot_count = np.zeros((n, n_slots), dtype=np.float32)

    for i, (formula, battery_id) in enumerate(zip(formulas, battery_ids)):
        try:
            pairs = order_elements_delithiation_first(formula, str(battery_id))[:n_slots]
        except Exception:
            continue

        for s, (element, count) in enumerate(pairs):
            slot_element_z[i, s] = int(Element(element).Z)
            slot_count[i, s] = float(count)

    return slot_element_z, slot_count
