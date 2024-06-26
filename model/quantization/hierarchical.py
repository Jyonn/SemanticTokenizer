from typing import List, Union, cast

import numpy as np
import torch
from pigmento import pnt
from torch import nn

from model.quantization.base import BaseQuantization
from model.quantization.multihead import MultiHeadQuantizationOutput
from model.quantization.vanilla import VanillaQuantization, VanillaQuantizationOutput


class HierarchicalQuantizationOutput(MultiHeadQuantizationOutput):
    pass


class HierarchicalQuantization(BaseQuantization):
    def __init__(
            self,
            num_codes: Union[int, List[int]],
            num_layers: int = None,
            threshold_disappearance=-1,
            **kwargs,
    ):
        super().__init__(**kwargs, num_codes=num_codes)

        if num_layers is None:
            assert isinstance(self.num_codes, list), "num_codes must be a list if num_layers is not specified"
            num_layers = len(self.num_codes)
        self.num_layers = num_layers

        if isinstance(self.num_codes, int):
            self.num_codes = [self.num_codes] * num_layers  # type: List[int]

        self.threshold_disappearance = threshold_disappearance

        self.quantizers = nn.ModuleList([
            VanillaQuantization(
                dim=self.embed_dim,
                num_codes=self.num_codes[i],
                commitment_cost=self.commitment_cost,
                threshold_disappearance=self.threshold_disappearance,
            )
            for i in range(self.num_layers)
        ])

    def initialize(
            self,
            embeds: torch.Tensor
    ):
        for i in range(self.num_layers):
            pnt(f'initialize quantizer {i} ...')
            quantizer = cast(VanillaQuantization, self.quantizers[i])
            quantizer.initialize(embeds)
            embeds = self.quantizers[i].codebook.weight.data

    def epoch_initialize(self):
        for i in range(self.num_layers):
            quantizer = cast(VanillaQuantization, self.quantizers[i])
            quantizer.epoch_initialize()

    # noinspection PyMethodMayBeStatic
    def get_next_layer_embeds(
            self,
            embeds,  # [B, D]
            layer_output: VanillaQuantizationOutput,
    ) -> torch.Tensor:
        return layer_output.embeds

    def quantize(
            self,
            embeds,  # [B, D]
            with_loss=False,
    ) -> HierarchicalQuantizationOutput:
        h_output = HierarchicalQuantizationOutput()
        if with_loss:
            h_output.loss = torch.tensor(0, dtype=torch.float, device=embeds.device)

        for i in range(self.num_layers):
            output = self.quantizers[i](embeds, with_loss=with_loss)
            embeds = self.get_next_layer_embeds(embeds, output)

            h_output.embeds.append(output.embeds)
            h_output.indices.append(output.indices)

            if with_loss:
                h_output.loss += output.loss

        return h_output

    def get_codebooks(self):
        codebooks = [q.get_codebooks() for q in self.quantizers]  # list[np.ndarray]
        return np.concatenate(codebooks, axis=0)
