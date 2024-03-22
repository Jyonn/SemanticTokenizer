import torch
from pigmento import pnt
from torch import nn
from vqtorch import ResidualVectorQuant


class ResidualQuantizationOutput:
    def __init__(
            self,
            embeds=None,
            indices=None,
            loss=None,
    ):
        self.embeds = embeds
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
        # pnt(f'RQ: groups={depth}, num_codes={num_codes}')

    def initialize(
            self,
            embeds: torch.Tensor
    ):
        pnt('initialize RQ ...')
        with torch.no_grad():
            self.quantizer(embeds)

    def __call__(
            self,
            embeds,  # [B, H, D]
            **kwargs,
    ) -> ResidualQuantizationOutput:
        quantized, output = self.quantizer(embeds)
        # for k in output:
        #     if output[k] is not None:
        #         pnt('output[%s]: %s' % (k, output[k].shape))
        #     else:
        #         pnt('output[%s]: None' % k)
        return ResidualQuantizationOutput(
            quantized,
            output['q'],
            output['loss'].mean(),
        )

    def get_codebooks(self):
        return self.quantizer.codebook.weight.data.cpu().numpy()
