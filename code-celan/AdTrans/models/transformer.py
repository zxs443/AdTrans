import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from .base import BaseModel
from .attention import AttentionPooling
from utils.config import DATASET_CONFIG, TRAINING_CONFIG, MODEL_CONFIG
from data.preprocessing import N_ELEMENTS
from utils.activation_utils import get_activation_module, get_functional_activation
from data.structure_placement import (
    get_structure_placement,
    get_structure_token_name,
    uses_structure_input,
    structure_in_encoder,
    structure_in_decoder,
    n_structure_tokens,
    structure_per_token_dim,
)

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        x = src
        if self.norm_first:
            x = self.norm1(x)

        need_weights = not self.training
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )

        x = x + self.dropout1(attn_output)
        if not self.norm_first:
            x = self.norm1(x)

        if self.norm_first:
            x = self.norm2(x)
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        if not self.norm_first:
            x = self.norm2(x)
        return x

class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None, tgt_is_causal: bool = False, memory_is_causal: bool = False) -> torch.Tensor:
        x = tgt
        if self.norm_first:
            x = self.norm1(x)

        need_weights = not self.training
        self_attn_output, self_attn_weights = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            is_causal=tgt_is_causal,
            need_weights=need_weights,
            average_attn_weights=False,
        )

        x = x + self.dropout1(self_attn_output)
        if not self.norm_first:
            x = self.norm1(x)

        if self.norm_first:
            x = self.norm2(x)
        cross_attn_output, cross_attn_weights = self.multihead_attn(
            x, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            is_causal=memory_is_causal,
            need_weights=need_weights,
            average_attn_weights=False,
        )

        x = x + self.dropout2(cross_attn_output)
        if not self.norm_first:
            x = self.norm2(x)

        if self.norm_first:
            x = self.norm3(x)
        x = x + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        if not self.norm_first:
            x = self.norm3(x)
        return x

class VoltagePredictor(BaseModel):
    """Encoder–decoder Transformer; optional CHGNet structure tokens."""
    def __init__(self, encoder_input_dim: int, d_model: int,
                 encoder_nhead: int, decoder_nhead: int, pooling_nhead: int,
                 num_layers: int, dropout_rate: float, output_hidden_layers: int,
                 decoder_input_dim: Optional[int] = None,
                 output_dims: Optional[list] = None,
                 decoder_token_names: Optional[list] = None,
                 n_property_tokens: Optional[int] = None,
                 structure_placement: Optional[str] = None,
                 structure_descriptor_dim: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim if decoder_input_dim is not None else DATASET_CONFIG.get('decoder_token_dim')
        if self.decoder_input_dim is None:
            raise ValueError("decoder_input_dim must be provided or defined in DATASET_CONFIG.")

        self.structure_placement = (
            structure_placement
            if structure_placement is not None
            else get_structure_placement()
        )
        self.structure_descriptor_dim = (
            structure_descriptor_dim
            if structure_descriptor_dim is not None
            else structure_per_token_dim()
        )

        self.output_hidden_layers = output_hidden_layers
        self.num_layers = num_layers
        self.decoder_token_names = decoder_token_names
        self.use_causal_mask_decoder = DATASET_CONFIG.get('use_causal_mask_decoder', False)

        self.n_property_tokens = self._resolve_n_property_tokens(
            decoder_token_names,
            n_property_tokens=n_property_tokens,
            structure_placement=self.structure_placement,
        )

        self.encoder_self_attention_weights = [[] for _ in range(num_layers)]
        self.decoder_self_attention_weights = [[] for _ in range(num_layers)]
        self.decoder_cross_attention_weights = [[] for _ in range(num_layers)]
        self.pooling_attention_weights = []

        self.model_activation_config = TRAINING_CONFIG['model_activation']
        model_activation_module = get_activation_module(self.model_activation_config)

        self.encoder_embed = nn.Sequential(
            nn.Linear(encoder_input_dim, d_model),
            model_activation_module,
        )

        self.decoder_embed = nn.Sequential(
            nn.Linear(self.decoder_input_dim, d_model),
            model_activation_module,
        )

        if uses_structure_input(self.structure_placement):
            self.structure_embed = nn.Sequential(
                nn.Linear(self.structure_descriptor_dim, d_model),
                model_activation_module,
            )
        else:
            self.structure_embed = None

        transformer_activation_fn = get_functional_activation(self.model_activation_config)
        self._initialize_output_dims(output_dims, d_model)

        encoder_layer = CustomTransformerEncoderLayer(
            d_model=d_model,
            nhead=encoder_nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=False,
            activation=transformer_activation_fn
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = CustomTransformerDecoderLayer(
            d_model=d_model,
            nhead=decoder_nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=False,
            activation=transformer_activation_fn
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.attention_pooling = AttentionPooling(
            d_model,
            nhead=pooling_nhead,
            dropout_rate=dropout_rate,
            activation=model_activation_module
        )

        layers = []
        current_dim = d_model
        for i in range(self.output_hidden_layers):
            if not self.output_dims:
                raise ValueError("output_dims list is empty, cannot build output layer.")
            next_dim = self.output_dims[i]
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(model_activation_module)
            layers.append(nn.LayerNorm(next_dim))
            current_dim = next_dim

        layers.append(nn.Linear(current_dim, 1))
        self.output_layer = nn.Sequential(*layers)

        self.n_element_slots = N_ELEMENTS
        self._register_attention_hooks()

    @staticmethod
    def _resolve_n_property_tokens(
        decoder_token_names: Optional[list],
        n_property_tokens: Optional[int] = None,
        structure_placement: Optional[str] = None,
    ) -> Optional[int]:
        if n_property_tokens is not None:
            return int(n_property_tokens)
        if not decoder_token_names:
            return None
        names = list(decoder_token_names)
        placement = structure_placement if structure_placement is not None else get_structure_placement()
        if structure_in_decoder(placement) and names:
            structure_name = get_structure_token_name()
            n_struct = n_structure_tokens(placement)
            if n_struct and names[-1] == structure_name:
                names = names[:-n_struct]
        return len(names)

    def _embed_structure_tokens(self, structure_input: torch.Tensor) -> torch.Tensor:
        if structure_input.dim() == 2:
            structure_input = structure_input.unsqueeze(1)
        return self.structure_embed(structure_input)

    def _build_encoder_sequence(
        self,
        encoder_input: torch.Tensor,
        structure_input: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x_enc = self.encoder_embed(encoder_input)
        if uses_structure_input(self.structure_placement):
            if structure_input is None:
                raise ValueError("structure_input must be provided when structure_placement is not 'none'.")
            x_struct = self._embed_structure_tokens(structure_input)
            if structure_in_encoder(self.structure_placement):
                x_enc = torch.cat([x_struct, x_enc], dim=1)
        return x_enc

    def _run_transformer_encoder(
        self,
        x_enc: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = x_enc
        for layer in self.transformer_encoder.layers:
            output = layer(
                output,
                src_mask=None,
                src_key_padding_mask=src_key_padding_mask,
            )
        if self.transformer_encoder.norm is not None:
            output = self.transformer_encoder.norm(output)
        return output

    def _initialize_output_dims(self, output_dims: Optional[list], d_model: int):
        if output_dims is not None:
            self.output_dims = output_dims
            if len(self.output_dims) != self.output_hidden_layers:
                raise ValueError(f"When output_dims is provided, its length ({len(self.output_dims)}) must match output_hidden_layers ({self.output_hidden_layers}).")
        else:
            self.output_dims = []
            if self.output_hidden_layers > 0:
                start_dim = d_model
                dims_linspace = np.linspace(start_dim, max(1, d_model // (self.output_hidden_layers + 1)), self.output_hidden_layers + 1)
                self.output_dims = [max(1, int(d)) for d in dims_linspace[1:]]

                for i in range(1, len(self.output_dims)):
                    if self.output_dims[i] >= self.output_dims[i-1]:
                        self.output_dims[i] = max(1, self.output_dims[i-1] - 1)
                        if self.output_dims[i] == 0 and self.output_dims[i-1] == 1:
                            self.output_dims[i] = 1

    def _save_attn_weights_hook(self, module, input, output, target_list):
        if self.training:
            return
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            return
        target_list.append(output[1].detach().cpu())

    def _register_attention_hooks(self):
        for i, layer in enumerate(self.transformer_encoder.layers):
            layer.self_attn.register_forward_hook(
                lambda m, i_in, o_out, layer_idx=i: self._save_attn_weights_hook(m, i_in, o_out, self.encoder_self_attention_weights[layer_idx])
            )

        for i, layer in enumerate(self.transformer_decoder.layers):
            layer.self_attn.register_forward_hook(
                lambda m, i_in, o_out, layer_idx=i: self._save_attn_weights_hook(m, i_in, o_out, self.decoder_self_attention_weights[layer_idx])
            )
            layer.multihead_attn.register_forward_hook(
                lambda m, i_in, o_out, layer_idx=i: self._save_attn_weights_hook(m, i_in, o_out, self.decoder_cross_attention_weights[layer_idx])
            )
        self.attention_pooling.attn.register_forward_hook(
            lambda m, i_in, o_out: self._save_attn_weights_hook(m, i_in, o_out, self.pooling_attention_weights)
        )

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                structure_input: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, list, list, list, list]:
        self.encoder_self_attention_weights = [[] for _ in range(self.num_layers)]
        self.decoder_self_attention_weights = [[] for _ in range(self.num_layers)]
        self.decoder_cross_attention_weights = [[] for _ in range(self.num_layers)]
        self.pooling_attention_weights = []

        x_enc = self._build_encoder_sequence(encoder_input, structure_input)
        x_dec = self.decoder_embed(decoder_input)

        if uses_structure_input(self.structure_placement):
            if structure_input is None:
                raise ValueError("structure_input must be provided when structure_placement is not 'none'.")
            x_struct = self._embed_structure_tokens(structure_input)
            if structure_in_decoder(self.structure_placement):
                x_dec = torch.cat([x_dec, x_struct], dim=1)

        tgt_mask = None
        if self.use_causal_mask_decoder:
            tgt_mask = self.generate_square_subsequent_mask(x_dec.size(1)).to(decoder_input.device)

        memory = self._run_transformer_encoder(
            x_enc,
            src_key_padding_mask=src_key_padding_mask,
        )

        output = self.transformer_decoder(x_dec, memory, tgt_mask, memory_key_padding_mask=src_key_padding_mask)
        pooled, pooling_attn_weights = self.attention_pooling(output)
        out = self.output_layer(pooled)
        return (
            out.squeeze(-1),
            self.encoder_self_attention_weights,
            self.decoder_self_attention_weights,
            self.decoder_cross_attention_weights,
            self.pooling_attention_weights,
        )

    def get_config(self) -> dict:
        return {
            'encoder_input_dim': self.encoder_input_dim,
            'd_model': self.d_model,
            'encoder_nhead': self.transformer_encoder.layers[0].self_attn.num_heads,
            'decoder_nhead': self.transformer_decoder.layers[0].self_attn.num_heads,
            'pooling_nhead': self.attention_pooling.attn.num_heads,
            'num_layers': len(self.transformer_encoder.layers),
            'dropout_rate': self.transformer_encoder.layers[0].dropout.p,
            'decoder_input_dim': self.decoder_input_dim,
            'decoder_token_names': self.decoder_token_names,
            'n_property_tokens': self.n_property_tokens,
            'structure_placement': self.structure_placement,
            'structure_descriptor_dim': self.structure_descriptor_dim,
            'output_hidden_layers': self.output_hidden_layers,
            'output_dims': self.output_dims,
            'model_activation': self.model_activation_config['type'],
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
