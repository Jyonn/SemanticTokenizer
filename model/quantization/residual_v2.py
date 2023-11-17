import torch
from torch import nn
from vqtorch import ResidualVectorQuant


class ResidualQuantizationOutput:
    def __init__(
            self,
            embeds,
            quantized,
            indices,
            loss,
    ):
        self.embeds = embeds
        self.quantized = quantized
        self.indices = indices
        self.loss = loss


class ResidualQuantizationV2(nn.Module):
    def __init__(
            self,
            num_codes: int,
            depth: int,
            **kwargs,
    ):
        super().__init__()
        self.quantizer = ResidualVectorQuant(
            num_codes=num_codes,
            groups=depth,
            share=False,
            **kwargs,
        )

    def initialize(
            self,
            embeds: torch.Tensor
    ):
        with torch.no_grad():
            self.quantizer(embeds)

    def __call__(
            self,
            embeds,  # [B, H, D]
    ) -> ResidualQuantizationOutput:
        quantized, output = self.quantizer(embeds)
        return ResidualQuantizationOutput(
            embeds,
            quantized,
            output['q'],
            output['loss'],
        )

    def get_codebooks(self):
        return self.quantizer.codebook.weight.data.cpu().numpy()
