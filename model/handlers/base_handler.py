import torch
from torch import nn

from loader.global_setting import Setting
from model.quantization.residual import ResidualQuantizationOutput


class HandlerOutput:
    def __init__(self, states, quantized, kl, recon):
        self.states = states
        self.quantized = quantized
        self.kl = kl
        self.recon = recon


class BaseHandler(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_empty_quantized_output():
        output = ResidualQuantizationOutput()
        output.loss = torch.tensor(0, dtype=torch.float).to(Setting.device)
        return output

    @staticmethod
    def get_zero_tensor():
        return torch.tensor(0, dtype=torch.float).to(Setting.device)

    def __call__(self, encoder_hidden_states) -> HandlerOutput:
        return HandlerOutput(
            states=encoder_hidden_states,
            quantized=self.get_empty_quantized_output(),
            kl=self.get_zero_tensor(),
            recon=self.get_zero_tensor(),
        )
