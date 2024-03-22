import torch
from torch import nn

from loader.global_setting import Setting
from model.handlers.base_handler import BaseHandler, HandlerOutput
from model.handlers.vae_handler import VAEHandler


class VAEQuantHandler(VAEHandler):
    def __call__(self, encoder_hidden_states):
        embeds = Setting.it.attention(encoder_hidden_states)
        mu, log_var, z = self.vae_encoder(embeds)

        quantized = self.quantizer(z, with_loss=True)
        quantized_embeds = torch.stack(quantized.embeds, dim=1)  # [B, H, D]

        states = self.vae_decoder(quantized_embeds)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return HandlerOutput(
            states=states,
            quantized=quantized,
            kl=kl_divergence,
            recon=torch.mean(torch.sum((embeds - states) ** 2, dim=-1)),
        )
