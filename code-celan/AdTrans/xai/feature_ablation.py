import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict
from data.dataset import unpack_batch, forward_model, voltage_from_model_output
from data.structure_placement import (
    structure_in_decoder,
    structure_in_encoder,
    get_structure_token_name,
    encoder_structure_offset,
    accumulate_structure_xai_importance,
    normalize_element_importance_excluding_structure,
)


class FeatureAblationExplainer:
    """Leave-one-out style ablation importance for element and property tokens."""
    def __init__(self, model: nn.Module, device: torch.device, ablation_value_type: str = 'zeros'):
        self.model = model
        self.device = device
        self.model.eval()
        self.ablation_value_type = ablation_value_type
        self.structure_token_name = get_structure_token_name()

    def _get_ablation_value_tensor(self, original_tensor: torch.Tensor) -> torch.Tensor:

        if self.ablation_value_type == 'zeros':
            return torch.zeros_like(original_tensor).to(self.device)
        if self.ablation_value_type == 'random':
            return torch.randn_like(original_tensor).to(self.device)
        raise ValueError(f"Unsupported ablation value type: {self.ablation_value_type}")

    def _prediction_delta(self, original_predictions: torch.Tensor, ablated_predictions: torch.Tensor) -> torch.Tensor:
        return torch.abs(original_predictions - ablated_predictions).view(-1)

    def _parse_batch_element_symbols(
        self, formulas_batch, battery_ids_batch
    ) -> list[list[str]]:
        from utils.element_slot_order import encoder_slot_element_symbols

        batch_symbols = []
        for idx, sample_formula in enumerate(formulas_batch):
            try:
                battery_id = str(battery_ids_batch[idx])
                batch_symbols.append(
                    encoder_slot_element_symbols(sample_formula, battery_id)
                )
            except Exception:
                print(f"Warning: Skip element ablation for sample: {sample_formula}")
                batch_symbols.append([])
        return batch_symbols

    def _ablate_elements_batch(
        self,
        enc_in_batch: torch.Tensor,
        dec_in_batch: torch.Tensor,
        src_mask_batch: torch.Tensor,
        structure_in_batch,
        original_predictions: torch.Tensor,
        batch_element_symbols: list[list[str]],
        element_importance_sum: dict,
        n_elements: int,
    ) -> None:
        for slot in range(n_elements):
            element_tensor_idx = encoder_structure_offset() + slot
            if element_tensor_idx >= enc_in_batch.size(1):
                continue

            active_samples = []
            active_symbols = []
            for sample_idx, symbols in enumerate(batch_element_symbols):
                if slot >= len(symbols):
                    continue
                if src_mask_batch[sample_idx, element_tensor_idx]:
                    continue
                active_samples.append(sample_idx)
                active_symbols.append(symbols[slot])

            if not active_samples:
                continue

            saved = enc_in_batch[:, element_tensor_idx, :].clone()
            try:
                enc_in_batch[:, element_tensor_idx, :] = self._get_ablation_value_tensor(saved)
                with torch.no_grad():
                    ablated_predictions = voltage_from_model_output(forward_model(
                        self.model, enc_in_batch, dec_in_batch, src_mask_batch, structure_in_batch,
                    ))
                deltas = self._prediction_delta(original_predictions, ablated_predictions)
                for sample_idx, symbol in zip(active_samples, active_symbols):
                    element_importance_sum[symbol] += deltas[sample_idx].item()
            finally:
                enc_in_batch[:, element_tensor_idx, :] = saved

    def _ablate_structure_batch(
        self,
        enc_in_batch: torch.Tensor,
        dec_in_batch: torch.Tensor,
        src_mask_batch: torch.Tensor,
        structure_in_batch,
        original_predictions: torch.Tensor,
        element_importance_sum: dict,
        property_importance_sum: dict,
    ) -> None:
        if structure_in_batch is None:
            return
        if not (structure_in_encoder() or structure_in_decoder()):
            return

        saved = structure_in_batch.clone()
        try:
            structure_in_batch.copy_(self._get_ablation_value_tensor(saved))
            with torch.no_grad():
                ablated_predictions = voltage_from_model_output(forward_model(
                    self.model, enc_in_batch, dec_in_batch, src_mask_batch, structure_in_batch,
                ))
            total_delta = self._prediction_delta(original_predictions, ablated_predictions).sum().item()
            accumulate_structure_xai_importance(
                total_delta,
                element_importance_sum,
                property_importance_sum,
                token_name=self.structure_token_name,
            )
        finally:
            structure_in_batch.copy_(saved)

    def _ablate_decoder_tokens_batch(
        self,
        enc_in_batch: torch.Tensor,
        dec_in_batch: torch.Tensor,
        src_mask_batch: torch.Tensor,
        structure_in_batch,
        original_predictions: torch.Tensor,
        decoder_token_names: list,
        property_importance_sum: dict,
    ) -> None:
        num_decoder_tokens = dec_in_batch.size(1)

        for token_idx, token_name in enumerate(decoder_token_names):
            if token_name == self.structure_token_name:
                continue
            if token_idx >= num_decoder_tokens:
                continue

            saved = dec_in_batch[:, token_idx, :].clone()
            try:
                dec_in_batch[:, token_idx, :] = self._get_ablation_value_tensor(saved)
                with torch.no_grad():
                    ablated_predictions = voltage_from_model_output(forward_model(
                        self.model, enc_in_batch, dec_in_batch, src_mask_batch, structure_in_batch,
                    ))
                property_importance_sum[token_name] += self._prediction_delta(
                    original_predictions, ablated_predictions
                ).sum().item()
            finally:
                dec_in_batch[:, token_idx, :] = saved

    def explain(self, train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                all_raw_formulas: list,
                n_elements: int,
                decoder_token_names: list,
                ablate_elements: bool = True,
                ablate_properties: bool = True,
                data_loaders=None):

        self.model.eval()

        element_importance_sum = defaultdict(float)
        property_importance_sum = defaultdict(float)

        if data_loaders is None:
            data_loaders = [train_loader, val_loader, test_loader]

        for data_loader in data_loaders:
            for batch in data_loader:
                unpacked = unpack_batch(batch, self.device)
                _, enc_in_batch, dec_in_batch, target_batch, src_mask_batch, formulas_batch, battery_ids_batch, structure_in_batch = unpacked

                with torch.no_grad():
                    original_predictions = voltage_from_model_output(forward_model(
                        self.model, enc_in_batch, dec_in_batch, src_mask_batch, structure_in_batch,
                    ))

                if ablate_elements:
                    batch_element_symbols = self._parse_batch_element_symbols(
                        formulas_batch, battery_ids_batch
                    )
                    self._ablate_elements_batch(
                        enc_in_batch,
                        dec_in_batch,
                        src_mask_batch,
                        structure_in_batch,
                        original_predictions,
                        batch_element_symbols,
                        element_importance_sum,
                        n_elements,
                    )

                if ablate_properties:
                    self._ablate_structure_batch(
                        enc_in_batch,
                        dec_in_batch,
                        src_mask_batch,
                        structure_in_batch,
                        original_predictions,
                        element_importance_sum,
                        property_importance_sum,
                    )
                    self._ablate_decoder_tokens_batch(
                        enc_in_batch,
                        dec_in_batch,
                        src_mask_batch,
                        structure_in_batch,
                        original_predictions,
                        decoder_token_names,
                        property_importance_sum,
                    )

        element_importance_df = normalize_element_importance_excluding_structure(
            pd.DataFrame(element_importance_sum.items(), columns=['Element', 'Importance'])
        )

        property_importance_df = pd.DataFrame(property_importance_sum.items(), columns=['Property', 'Importance'])
        if not property_importance_df.empty:
            total_sum = property_importance_df['Importance'].sum()
            if total_sum > 0:
                property_importance_df['Importance'] = property_importance_df['Importance'] / total_sum
            else:
                property_importance_df['Importance'] = 0.0
            property_importance_df = property_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

        return element_importance_df, property_importance_df
