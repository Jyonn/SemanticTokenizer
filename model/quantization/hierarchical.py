from typing import List, Union

import torch
from torch import nn

from model.quantization.base import BaseQuantization
from model.quantization.multihead import MultiHeadQuantizationOutput
from model.quantization.vanilla import VanillaQuantization, VanillaQuantizationOutput


class HierarchicalQuantizationOutput(MultiHeadQuantizationOutput):
    pass


class HierarchicalQuantization(BaseQuantization):
    def __init__(
            self,
            num_layers: int,
            num_codes: Union[int, List[int]],
            **kwargs,
    ):
        super().__init__(**kwargs, num_codes=num_codes)

        if isinstance(self.num_codes, int):
            self.num_codes = [self.num_codes] * num_layers  # type: List[int]

        self.quantizers = nn.ModuleList([
            VanillaQuantization(
                dim=self.embed_dim,
                num_codes=self.num_codes[i],
                commitment_cost=self.commitment_cost,
            )
            for i in range(self.num_heads)
        ])  # type: nn.ModuleList[VanillaQuantization]

    def initialize(
            self,
            embeds: torch.Tensor
    ):
        for i in range(self.num_heads):
            self.quantizers[i].initialize(embeds)
            embeds = self.quantizers[i].codebook.weight.data

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

        for i in range(self.num_heads):
            output = self.quantizers[i](embeds, with_loss=with_loss)
            embeds = self.get_next_layer_embeds(embeds, output)

            h_output.embeds.append(output.embeds)
            h_output.indices.append(output.indices)

            if with_loss:
                h_output.loss += output.loss

        return h_output
