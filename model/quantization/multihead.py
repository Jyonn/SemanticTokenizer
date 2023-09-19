import torch
from torch import nn

from model.quantization.base import BaseQuantization, BaseQuantizationOutput
from model.quantization.vanilla import VanillaQuantization


class MultiHeadQuantizationOutput(BaseQuantizationOutput):
    def __init__(self):
        super().__init__(
            embeds=[],
            indices=[],
        )


class MultiHeadQuantization(BaseQuantization):
    """
    Multi-head quantization, also known as product quantization
    """
    def __init__(
            self,
            num_heads: int,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_heads = num_heads

        self.quantizers = nn.ModuleList([
            VanillaQuantization(
                dim=self.embed_dim,
                num_codes=self.num_codes,
                commitment_cost=self.commitment_cost,
            )
            for _ in range(self.num_heads)
        ])  # type: nn.ModuleList[BaseQuantization]

    def quantize(
            self,
            embeds,  # [B, H, D]
            with_loss=False,
    ) -> MultiHeadQuantizationOutput:
        mh_output = MultiHeadQuantizationOutput()
        if with_loss:
            mh_output.loss = torch.tensor(0, dtype=torch.float, device=embeds.device)

        for i in range(self.num_heads):
            output = self.quantizers[i](embeds[:, i, :], with_loss=with_loss)
            mh_output.embeds.append(output.embeds)
            mh_output.indices.append(output.indices)

            if with_loss:
                mh_output.loss += output.loss

        return mh_output
