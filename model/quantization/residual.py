import torch

from model.quantization.hierarchical import HierarchicalQuantizationOutput, HierarchicalQuantization
from model.quantization.vanilla import VanillaQuantizationOutput


class ResidualQuantizationOutput(HierarchicalQuantizationOutput):
    pass


class ResidualQuantization(HierarchicalQuantization):
    def initialize(
            self,
            embeds: torch.Tensor
    ):
        for i in range(self.num_heads):
            self.quantizers[i].initialize(embeds)
            centers = self.quantizers[i].codebook.weight.data
            dist = torch.cdist(embeds, centers, p=2)
            embeds = embeds - centers[dist.argmin(dim=1)]

    def get_next_layer_embeds(
            self,
            embeds,  # [B, D]
            layer_output: VanillaQuantizationOutput,
    ) -> torch.Tensor:
        return embeds - layer_output.embeds
