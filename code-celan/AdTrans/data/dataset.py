import numpy as np
import torch
from torch.utils.data import Dataset
from data.structure_placement import uses_structure_input


class MaterialsDataset(Dataset):
    """MaterialsDataset: encoder/decoder tensors + voltage targets."""

    def __init__(
        self,
        encoder_data,
        decoder_data,
        targets,
        src_key_padding_mask,
        formulas,
        battery_ids,
        structure_data=None,
    ):
        self.encoder_data = torch.tensor(encoder_data, dtype=torch.float32)
        self.decoder_data = torch.tensor(decoder_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.src_key_padding_mask = torch.tensor(src_key_padding_mask, dtype=torch.bool)
        self.formulas = formulas
        self.battery_ids = np.asarray(battery_ids, dtype=object)
        if len(self.battery_ids) != len(self.targets):
            raise ValueError(
                f"battery_ids length ({len(self.battery_ids)}) must match targets length ({len(self.targets)})."
            )
        self.structure_data = (
            torch.tensor(structure_data, dtype=torch.float32) if structure_data is not None else None
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        base = (
            idx,
            self.encoder_data[idx],
            self.decoder_data[idx],
            self.targets[idx],
            self.src_key_padding_mask[idx],
            self.formulas[idx],
            self.battery_ids[idx],
        )
        if self.structure_data is not None:
            base = base + (self.structure_data[idx],)
        return base


def unpack_batch(batch, device=None):
    """Unpack batch; optional `.to(device)`."""
    idx = 0
    batch_idx = batch[idx]
    idx += 1
    enc_in = batch[idx]
    idx += 1
    dec_in = batch[idx]
    idx += 1
    target = batch[idx]
    idx += 1
    src_mask = batch[idx]
    idx += 1
    formulas = batch[idx]
    idx += 1
    battery_ids = batch[idx]
    idx += 1

    structure_in = None
    if uses_structure_input():
        structure_in = batch[idx]
        idx += 1

    if device is not None:
        enc_in = enc_in.to(device)
        dec_in = dec_in.to(device)
        target = target.to(device)
        src_mask = src_mask.to(device)
        if structure_in is not None:
            structure_in = structure_in.to(device)

    return (
        batch_idx,
        enc_in,
        dec_in,
        target,
        src_mask,
        formulas,
        battery_ids,
        structure_in,
    )


def voltage_from_model_output(model_out):
    return model_out[0]


def forward_model(model, enc_in, dec_in, src_mask, structure_in=None):
    kwargs = {'src_key_padding_mask': src_mask}
    if uses_structure_input():
        kwargs['structure_input'] = structure_in
    return model(enc_in, dec_in, **kwargs)
