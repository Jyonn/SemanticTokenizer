import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from loader.global_setting import Setting
from model.quantization.base import BaseQuantization, BaseQuantizationOutput


class VanillaQuantizationOutput(BaseQuantizationOutput):
    pass


class VanillaQuantization(BaseQuantization):
    def __init__(
            self,
            threshold_disappearance=-1,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.codebook = nn.Embedding(self.num_codes, self.embed_dim)
        self.codecount = np.zeros(self.num_codes, dtype=np.int32)

        self.threshold_disappearance = threshold_disappearance
        self.num_quantized = 0

    def initialize(self, embeds: torch.Tensor):
        """
        use k-means to initialize the codebook based on the entire embedding space
        :param embeds: [N, D]
        """
        kmeans = KMeans(n_clusters=self.num_codes, random_state=0).fit(embeds.cpu().numpy())
        self.codebook.weight.data.copy_(torch.from_numpy(kmeans.cluster_centers_))

    def update(
            self,
            embeds,
            indices,
    ):
        self.num_quantized += embeds.shape[0]
        self.codecount += np.bincount(indices.cpu().numpy().reshape(-1), minlength=self.num_codes)

        # if current code is not used by any entity, reinitialize it
        mask = self.num_quantized // self.num_codes < self.codecount * self.threshold_disappearance
        mask = torch.from_numpy(mask).to(embeds.device)

        if torch.any(mask):
            # randomly sample embeds
            indices = np.arange(embeds.shape[0])
            indices = np.random.choice(indices, size=mask.sum(), replace=False)
            embeds = embeds[indices]

            # replace masked codebook with random embeddings
            self.codebook.weight.data[mask] = embeds

            # reset codecount
            self.num_quantized = 0
            self.codecount = np.zeros(self.num_codes, dtype=np.int32)

    def quantize(
            self,
            embeds,  # [B, D]
            with_loss=False,
    ) -> VanillaQuantizationOutput:
        dist = torch.cdist(embeds, self.codebook.weight, p=2)
        indices = torch.argmin(dist, dim=-1).unsqueeze(1)

        if self.training and self.threshold_disappearance > 0:
            self.update(embeds, indices)

        ph = torch.zeros(indices.shape[0], self.num_codes, device=embeds.device)  # [B, C]
        ph.scatter_(1, indices, 1)
        codes = torch.matmul(ph, self.codebook.weight).view(embeds.shape)

        output = VanillaQuantizationOutput(codes, indices=indices)

        if not with_loss:
            return output

        loss = torch.tensor(0, dtype=torch.float, device=embeds.device)
        loss += F.mse_loss(codes.detach(), embeds) * self.commitment_cost + F.mse_loss(codes, embeds.detach())
        return output.set_loss(loss)
