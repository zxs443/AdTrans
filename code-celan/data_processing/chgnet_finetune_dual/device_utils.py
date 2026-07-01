"""Resolve PyTorch device with GPU fallback."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_MIN_CUDA_CAPABILITY = (3, 7)


def cuda_is_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) < _MIN_CUDA_CAPABILITY:
            return False
        probe = torch.zeros(1, device="cuda")
        probe.add_(1)
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def resolve_torch_device(
    requested: str = "auto",
    *,
    log: bool = True,
) -> torch.device:
    req = (requested or "auto").strip().lower()
    if req in {"auto", "cuda"}:
        if cuda_is_usable():
            device = torch.device("cuda")
            if log:
                name = torch.cuda.get_device_name(0)
                cap = torch.cuda.get_device_capability(0)
                logger.info("Using CUDA device: %s (capability %s.%s)", name, cap[0], cap[1])
            return device
        if log:
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(0)
                name = torch.cuda.get_device_name(0)
                logger.warning(
                    "CUDA device %s (capability %s.%s) is not supported by this PyTorch build; using CPU.",
                    name,
                    cap[0],
                    cap[1],
                )
            elif req == "cuda":
                logger.warning("CUDA requested but not available; using CPU.")
            else:
                logger.info("CUDA not usable; using CPU.")
        return torch.device("cpu")

    if req == "cpu":
        return torch.device("cpu")

    if req.startswith("cuda"):
        if cuda_is_usable():
            return torch.device(requested)
        if log:
            logger.warning("Requested device %s is not usable; using CPU.", requested)
        return torch.device("cpu")

    if log:
        logger.warning("Unknown device %r; using CPU.", requested)
    return torch.device("cpu")


def resolve_device_name(requested: str = "auto", *, log: bool = True) -> str:
    return str(resolve_torch_device(requested, log=log))


def resolve_preload_device(
    requested: str,
    train_device: torch.device,
    *,
    log: bool = True,
) -> torch.device:
    req = (requested or "auto").strip().lower()
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if train_device.type != "cuda":
            raise ValueError("preload_device='cuda' requires CUDA training device")
        return train_device
    if req == "auto":
        resolved = train_device if train_device.type == "cuda" else torch.device("cpu")
        if log and resolved.type == "cuda":
            logger.info("preload_device=auto -> CUDA graph preload enabled")
        return resolved
    raise ValueError(f"Unknown preload_device: {requested!r}")


def crystal_graph_device(graph) -> torch.device | None:
    for attr in ("atomic_numbers", "atom_graph", "bond_graph", "natoms"):
        val = getattr(graph, attr, None)
        if isinstance(val, torch.Tensor):
            return val.device
    return None


def move_graphs_to_device(graphs, device: torch.device | str):
    target = torch.device(device)
    moved = []
    for graph in graphs:
        current = crystal_graph_device(graph)
        if current is not None and current == target:
            moved.append(graph)
        else:
            moved.append(graph.to(target))
    return moved
