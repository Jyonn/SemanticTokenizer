import torch

from loader.global_setting import Setting
from model.handlers.base_handler import BaseHandler, HandlerOutput


class QuantHandler(BaseHandler):
    def __call__(self, encoder_hidden_states):
        embeds = Setting.it.attention(encoder_hidden_states)
        quantized = Setting.it.quantizer(embeds, with_loss=True)
        states = torch.sum(torch.stack(quantized.embeds, dim=-1), dim=-1)

        return HandlerOutput(
            states=states,
            quantized=quantized,
            kl=self.get_zero_tensor(),
            recon=torch.mean(torch.sum((embeds - states) ** 2, dim=-1)),
        )
