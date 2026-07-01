from pymatgen.core import Composition

_SUB_SUP_TO_ASCII = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
}

def _normalize_formula(formula: str) -> str:
    if not isinstance(formula, str):
        return formula

    normalized_formula = []
    for char in formula:
        normalized_formula.append(_SUB_SUP_TO_ASCII.get(char, char))
    return "".join(normalized_formula)

def _parse_formula_composition(formula):

    normalized_formula = _normalize_formula(formula)
    try:
        return Composition(normalized_formula)
    except Exception as e:
        print(f"Warning: Error parsing chemical formula '{formula}' (normalized: '{normalized_formula}'): {e}.")
        return None

def _get_element_symbols_from_formula(formula):
    """Element symbols from formula (pymatgen)."""
    comp = _parse_formula_composition(formula)
    if comp:
        return [str(el) for el in comp.elements]
    return []

def _count_unique_elements_in_formula(formula, n_elements):
    """Count distinct elements in a formula (used for padding-mask sizing; not capped here)."""
    comp = _parse_formula_composition(formula)
    if comp:
        return len(comp.elements)
    print(f"Warning: Could not parse formula '{formula}'. Assuming {n_elements} elements for mask generation.")
    return n_elements 
