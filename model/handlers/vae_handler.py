import torch
from torch import nn

from loader.global_setting import Setting
from model.handlers.base_handler import BaseHandler, HandlerOutput


class VAEEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_mu = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_var = nn.Linear(self.embed_dim, self.embed_dim)

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        log_var = self.fc_var(h1)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z


class VAEDecoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc2(h1)


class VAEHandler(BaseHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.vae_encoder = VAEEncoder(Setting.it.config.embed_dim)
        self.vae_decoder = VAEDecoder(Setting.it.config.embed_dim)

    def __call__(self, encoder_hidden_states):
        embeds = Setting.it.attention(encoder_hidden_states)
        mu, log_var, z = self.vae_encoder(embeds)
        states = self.vae_decoder(z)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return HandlerOutput(
            states=states,
            quantized=self.get_empty_quantized_output(),
            kl=kl_divergence,
            recon=self.get_zero_tensor(),
        )
