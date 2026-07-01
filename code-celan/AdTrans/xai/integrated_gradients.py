import torch
import torch.nn as nn
import numpy as np
from data.dataset import forward_model, voltage_from_model_output

class IntegratedGradientsExplainer:
    """Integrated Gradients for encoder, decoder, and optional CHGNet structure inputs."""
    def __init__(self, model: nn.Module, device: torch.device, baseline_type: str = 'zeros',
                 encoder_mean: np.ndarray = None, decoder_mean: np.ndarray = None,
                 structure_mean: np.ndarray = None, inputs_are_standardized: bool = True):
        self.model = model
        self.device = device
        self.model.eval()
        self.baseline_type = baseline_type
        self.inputs_are_standardized = inputs_are_standardized
        self.encoder_mean = torch.tensor(encoder_mean, dtype=torch.float32).to(self.device) if encoder_mean is not None else None
        self.decoder_mean = torch.tensor(decoder_mean, dtype=torch.float32).to(self.device) if decoder_mean is not None else None
        self.structure_mean = torch.tensor(structure_mean, dtype=torch.float32).to(self.device) if structure_mean is not None else None

    def _get_baseline(self, input_tensor: torch.Tensor, input_type: str = 'encoder') -> torch.Tensor:
        """Baseline tensor (zeros or train mean in input space)."""
        if self.baseline_type == 'zeros':
            return torch.zeros_like(input_tensor).to(self.device)

        if self.baseline_type == 'mean':
            # Training data are z-scored before entering the model; the training mean baseline is zero.
            if self.inputs_are_standardized:
                return torch.zeros_like(input_tensor).to(self.device)

            if input_type == 'encoder':
                if self.encoder_mean is None:
                    raise ValueError("encoder_mean must be provided when baseline_type is 'mean' and inputs are not standardized.")
                return self.encoder_mean.unsqueeze(0).expand_as(input_tensor)
            if input_type == 'decoder':
                if self.decoder_mean is None:
                    raise ValueError("decoder_mean must be provided when baseline_type is 'mean' and inputs are not standardized.")
                return self.decoder_mean.unsqueeze(0).expand_as(input_tensor)
            if input_type == 'structure':
                if self.structure_mean is None:
                    raise ValueError("structure_mean must be provided when baseline_type is 'mean' and inputs are not standardized.")
                return self.structure_mean.unsqueeze(0).expand_as(input_tensor)

        raise ValueError(f"Unsupported baseline type: {self.baseline_type}")

    def attribute(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor,
                  src_key_padding_mask: torch.Tensor, steps: int = 50,
                  structure_input: torch.Tensor = None):
        self.model.eval()

        encoder_input = encoder_input.float().to(self.device)
        decoder_input = decoder_input.float().to(self.device)
        src_key_padding_mask = src_key_padding_mask.to(self.device)
        if structure_input is not None:
            structure_input = structure_input.float().to(self.device)

        encoder_baseline = self._get_baseline(encoder_input, input_type='encoder')
        decoder_baseline = self._get_baseline(decoder_input, input_type='decoder')
        structure_baseline = None
        if structure_input is not None:
            structure_baseline = self._get_baseline(structure_input, input_type='structure')

        encoder_delta = encoder_input - encoder_baseline
        decoder_delta = decoder_input - decoder_baseline
        structure_delta = structure_input - structure_baseline if structure_input is not None else None

        encoder_attributions = torch.zeros_like(encoder_input).to(self.device)
        decoder_attributions = torch.zeros_like(decoder_input).to(self.device)
        structure_attributions = (
            torch.zeros_like(structure_input).to(self.device) if structure_input is not None else None
        )

        for i in range(steps + 1):
            alpha = float(i) / steps

            interpolated_encoder_input = (
                encoder_baseline + alpha * encoder_delta
            ).detach().requires_grad_(True)
            interpolated_decoder_input = (
                decoder_baseline + alpha * decoder_delta
            ).detach().requires_grad_(True)
            interpolated_structure_input = None
            if structure_input is not None:
                interpolated_structure_input = (
                    structure_baseline + alpha * structure_delta
                ).detach().requires_grad_(True)

            output = voltage_from_model_output(forward_model(
                self.model,
                interpolated_encoder_input,
                interpolated_decoder_input,
                src_key_padding_mask,
                interpolated_structure_input,
            ))

            grad_inputs = [interpolated_encoder_input, interpolated_decoder_input]
            if interpolated_structure_input is not None:
                grad_inputs.append(interpolated_structure_input)

            # One backward/step; grad w.r.t. sum(output) matches per-sample stack
            grads = torch.autograd.grad(
                output.sum(),
                grad_inputs,
                retain_graph=False,
                allow_unused=True,
            )

            encoder_attributions += grads[0].detach()
            decoder_attributions += grads[1].detach()
            if structure_attributions is not None and len(grads) > 2 and grads[2] is not None:
                structure_attributions += grads[2].detach()

        encoder_attributions = encoder_attributions * encoder_delta / steps
        decoder_attributions = decoder_attributions * decoder_delta / steps
        if structure_attributions is not None:
            structure_attributions = structure_attributions * structure_delta / steps

        return encoder_attributions, decoder_attributions, structure_attributions
