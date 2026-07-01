import inspect

import torch
import torch.nn as nn


def _init_kwargs_for_class(cls, config: dict) -> dict:
    """Keep only keyword arguments accepted by ``cls.__init__``."""
    params = inspect.signature(cls.__init__).parameters
    return {key: value for key, value in config.items() if key in params and key != "self"}


class BaseModel(nn.Module):
    """Shared utilities for Transformer regression models (masking, save/load)."""

    def __init__(self):
        super().__init__()

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = torch.nn.Transformer.generate_square_subsequent_mask(sz)
        return mask

    def save_model(self, path: str, config: dict = None):
        save_data = {
            'state_dict': self.state_dict()
        }
        if config is not None:
            save_data['config'] = config
        torch.save(save_data, path)

    @classmethod
    def load_model(cls, path: str, device: torch.device, **kwargs):
        checkpoint = torch.load(path, map_location=device)

        model_config = checkpoint.get('config', None)
        if model_config is not None:
            merged_config = model_config.copy()
            for key, value in kwargs.items():
                if key not in merged_config:
                    merged_config[key] = value
            final_kwargs = _init_kwargs_for_class(cls, merged_config)
            model = cls(**final_kwargs).to(device)
        else:
            if not kwargs:
                raise ValueError(
                    "Model configuration not saved; provide model instantiation parameters via kwargs."
                )
            model = cls(**kwargs).to(device)

        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=True)
        if missing:
            raise RuntimeError(f"load_model missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            raise RuntimeError(f"load_model unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        model.eval()
        return model
