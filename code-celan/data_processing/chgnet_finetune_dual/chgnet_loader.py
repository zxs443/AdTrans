from __future__ import annotations

import os
from typing import Optional, Tuple

from chgnet.model.model import CHGNet

from .device_utils import resolve_device_name

_model_singleton: Optional[CHGNet] = None
_model_sig: Optional[Tuple[Optional[str], str, Optional[str]]] = None
DEFAULT_CHGNET_COLUMN_PREFIX = "chg_struct_emb"


def _resolve_chgnet_device(explicit: Optional[str]) -> Optional[str]:
    if explicit is not None and str(explicit).strip():
        requested = str(explicit).strip().lower()
    else:
        env = os.environ.get("CHGNET_DEVICE", "").strip()
        requested = env if env else "auto"
    return resolve_device_name(requested, log=True)


def get_chgnet_model(
    use_device: Optional[str] = "cpu",
    model_name: str = "0.3.0",
    checkpoint_path: Optional[str] = None,
) -> CHGNet:
    global _model_singleton, _model_sig
    dev_key = _resolve_chgnet_device(use_device if use_device is not None else "auto")
    ckpt_key = str(checkpoint_path) if checkpoint_path else None
    sig = (dev_key, str(model_name), ckpt_key)
    if _model_singleton is not None and _model_sig == sig:
        return _model_singleton

    if checkpoint_path:
        _model_singleton = CHGNet.from_file(checkpoint_path)
    else:
        _model_singleton = CHGNet.load(
            model_name=model_name,
            use_device=dev_key,
        )

    _model_singleton = _model_singleton.to(dev_key)
    _model_sig = sig
    return _model_singleton


def get_chgnet_graph_converter(model_name: str = "0.3.0"):
    return get_chgnet_model(use_device="cpu", model_name=model_name).graph_converter
