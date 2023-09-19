import torch
from torch import nn
from torch.nn import functional as F

from model.quantization.base import BaseQuantization, BaseQuantizationOutput


class VanillaQuantizationOutput(BaseQuantizationOutput):
    pass


class VanillaQuantization(BaseQuantization):
    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.codebook = nn.Embedding(self.num_codes, self.embed_dim)

    def quantize(
            self,
            embeds,  # [B, D]
            with_loss=False,
    ) -> VanillaQuantizationOutput:
        dist = torch.cdist(embeds, self.codebooks.weight, p=2)
        indices = torch.argmin(dist, dim=-1).unsqueeze(1)
        ph = torch.zeros(indices.shape[0], self.num_codes, device=embeds.device)  # [B, C]
        ph.scatter_(1, indices, 1)
        codes = torch.matmul(ph, self.codebook.weight).view(embeds.shape)

        output = VanillaQuantizationOutput(codes, indices=indices)

        if not with_loss:
            return output

        loss = torch.tensor(0, dtype=torch.float, device=embeds.device)
        loss += F.mse_loss(codes.detach(), embeds) * self.commitment_cost + F.mse_loss(codes, embeds.detach())
        return output.set_loss(loss)
