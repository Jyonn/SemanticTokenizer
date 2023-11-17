from typing import Union, List, cast

import torch
from pigmento import pnt
from torch import nn
from vqtorch import VectorQuant, GroupVectorQuant

from model.quantization.base import BaseQuantization, BaseQuantizationOutput
from model.quantization.vanilla import VanillaQuantization


class MultiHeadQuantizationOutput:
    def __init__(
            self,
    ):
        self.embeds = []
        self.quantized = []
        self.indices = []
        self.loss = []

    def update(self, embeds, quantized, indices, loss):
        self.embeds.append(embeds)
        self.quantized.append(quantized)
        self.indices.append(indices)
        self.loss.append(loss)


class MultiHeadQuantizationV2(nn.Module):
    def __init__(
            self,
            num_codes: Union[int, List[int]],
            num_heads: int = None,
            **kwargs,
    ):
        super().__init__()

        if num_heads is None:
            assert isinstance(num_codes, list), 'num_codes must be a list if num_heads is not specified'
            num_heads = len(num_codes)
        if isinstance(num_codes, int):
            num_codes = [num_codes] * num_heads

        self.num_heads = num_heads
        self.num_codes = num_codes

        self.quantizers = nn.ModuleList([
            VectorQuant(
                num_codes=num_codes[i],
                **kwargs,
            )
            for i in range(self.num_heads)
        ])  # type: nn.ModuleList[BaseQuantization]

    def initialize(
            self,
            embeds: torch.Tensor
    ):
        with torch.no_grad():
            for i in range(self.num_heads):
                embed = embeds[:, i, :]
                self.quantizers[i](embed)

    def __call__(
            self,
            embeds,  # [B, H, D]
    ) -> MultiHeadQuantizationOutput:
        mh_output = MultiHeadQuantizationOutput()

        for i in range(self.num_heads):
            embed = embeds[:, i, :]
            quantized, output = self.quantizers[i](embed)
            mh_output.update(embeds[:, i, :], quantized, output['q'], output['loss'])
        mh_output.loss = torch.stack(mh_output.loss).mean()
        return mh_output

    def get_codebooks(self):
        codebooks = []
        for i in range(self.num_heads):
            quantizer = cast(VectorQuant, self.quantizers[i])
            codebooks.append(quantizer.codebook.weight.data.cpu().numpy())
        return codebooks
