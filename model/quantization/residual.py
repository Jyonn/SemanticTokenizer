import torch

from model.quantization.hierarchical import HierarchicalQuantizationOutput, HierarchicalQuantization
from model.quantization.vanilla import VanillaQuantizationOutput


class ResidualQuantizationOutput(HierarchicalQuantizationOutput):
    pass


class ResidualQuantization(HierarchicalQuantization):
    def get_next_layer_embeds(
            self,
            embeds,  # [B, D]
            layer_output: VanillaQuantizationOutput,
    ) -> torch.Tensor:
        return embeds - layer_output.embeds
