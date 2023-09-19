from torch import nn


class BaseQuantizationOutput:
    """
    Base class for quantization output
    """
    def __init__(self, embeds, indices=None):
        super().__init__()

        self.embeds = embeds
        self.indices = indices
        self.loss = None

    def set_loss(self, loss):
        self.loss = loss
        return self


class BaseQuantization(nn.Module):
    """
    Base class for quantization, supporting differentiable quantization
    """
    def __init__(
            self,
            dim,
            num_codes: int,
            commitment_cost=0.0,
    ):
        super().__init__()

        self.embed_dim = dim
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost

    def quantize(
            self,
            embeds,
            with_loss=False,
    ) -> BaseQuantizationOutput:
        pass

    def __call__(self, *args, **kwargs):
        return self.quantize(*args, **kwargs)
