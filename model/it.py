import torch
from pigmento import pnt
from torch import nn
from transformers import BartModel
from transformers.modeling_outputs import BaseModelOutput

from loader.embedding.embedding_manager import EmbeddingManager
from loader.global_setting import Setting
from model.inputer.concat_inputer import ConcatInputer
from model.quantization.multihead_v2 import MultiHeadQuantizationV2, MultiHeadQuantizationOutput
from model.quantization.residual_v2 import ResidualQuantizationV2
from model.utils.attention import AdditiveAttention
from model.utils.base_classifier import BaseClassifier


class ITOutput:
    def __init__(
            self,
            last_hidden_states: torch.Tensor,
            quantization_loss: torch.Tensor,
            generation_loss: torch.Tensor,
            reconstruction_loss: torch.Tensor,
            kl_divergence: torch.Tensor,
            voc_loss: dict,
            pred_labels: torch.Tensor = None,
            true_labels: torch.Tensor = None,
    ):
        self.last_hidden_states = last_hidden_states
        self.pred_labels = pred_labels
        self.true_labels = true_labels

        self.quantization_loss = quantization_loss
        self.generation_loss = generation_loss
        self.reconstruction_loss = reconstruction_loss
        self.kl_divergence = kl_divergence
        self.voc_loss = voc_loss


class ITConfig:
    def __init__(
            self,
            embed_dim: int = 768,
            hidden_size: int = 256,
            num_heads: int = 3,
            num_codes: int = 512,
            residual_depth: int = 4,
            commitment_cost: float = 0.0,
            skip_quant: bool = False,
            low_rank: bool = False,
            vae: bool = False,
            vae_quant: bool = False,
            kd: bool = False,
            **kwargs,
    ):
        if isinstance(num_codes, str):
            num_codes = eval(num_codes)

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.residual_depth = residual_depth
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost
        self.skip_quant = skip_quant
        self.low_rank = low_rank
        self.vae = vae
        self.vae_quant = vae_quant
        self.kd = kd


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


class IT(nn.Module):
    def __init__(
            self,
            config: ITConfig,
            inputer: ConcatInputer,
            embedding_manager: EmbeddingManager,
    ):
        super().__init__()

        self.config = config
        self.inputer = inputer
        self.depot = inputer.depot
        self.max_sequence_len = inputer.max_sequence_len

        self.embedding_manager = embedding_manager
        self.embedding_table = embedding_manager.get_table()

        self.bart = BartModel.from_pretrained('facebook/bart-base')  # type: BartModel

        self.back_outer_proj = nn.Linear(self.max_sequence_len, self.config.num_heads)
        # self.back_inner_proj = nn.Linear(self.config.embed_dim, self.config.hidden_size)
        self.back_inner_proj = lambda x: x
        self.vae_encoder = VAEEncoder(self.config.embed_dim)

        # self.forth_inner_proj = nn.Linear(self.config.hidden_size, self.config.embed_dim)
        self.forth_inner_proj = lambda x: x
        self.forth_outer_proj = nn.Linear(self.config.num_heads, self.max_sequence_len)
        self.vae_decoder = VAEDecoder(self.config.embed_dim)

        self.attention = AdditiveAttention(self.config.embed_dim, self.config.hidden_size)

        if not self.config.skip_quant:
            # self.quantizer = MultiHeadQuantizationV2(
            #     feature_size=self.config.embed_dim,
            #     num_heads=self.config.num_heads,
            #     num_codes=self.config.num_codes,
            #     beta=self.config.commitment_cost,
            #     kmeans_init=True,
            #     affine_lr=10.0,
            #     replace_freq=10,
            #     sync_nu=0.2,
            #     dim=-1,
            # )

            self.quantizer = ResidualQuantizationV2(
                feature_size=self.config.embed_dim,
                num_codes=self.config.num_codes,
                depth=self.config.residual_depth,
                beta=self.config.commitment_cost,
                kmeans_init=True,
                affine_lr=10.0,
                replace_freq=10,
                sync_nu=0.2,
                dim=-1,
            )

        self.classifier = self.embedding_manager.get_classifier()
        self.loss_fct = nn.CrossEntropyLoss()

        if self.config.kd:
            # frozen bart and classifier
            for param in self.bart.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False

        self.handler = self.get_handler()

    def get_handler(self):
        if self.config.skip_quant:
            return self.skip_quant
        if self.config.low_rank:
            return self.low_rank
        if self.config.vae:
            return self.vae
        if self.config.vae_quant:
            return self.vae_quant
        return self.quant

    def _encode(self, batch: dict):
        encoder_attention_mask = self.inputer.get_mask(batch['encoder'])
        encoder_input_embeddings = self.inputer.get_embeddings(batch['encoder'])

        output: BaseModelOutput = self.bart.encoder(
            inputs_embeds=encoder_input_embeddings,
            attention_mask=encoder_attention_mask,
            return_dict=True
        )
        encoder_hidden_states = output.last_hidden_state  # [B, L, D]
        return encoder_hidden_states

    def _decode(self, batch, input_hidden_states):
        encoder_attention_mask = self.inputer.get_mask(batch['encoder'])
        decoder_attention_mask = self.inputer.get_mask(batch['decoder'])
        decoder_input_embeddings = self.inputer.get_embeddings(batch['decoder'])

        output: BaseModelOutput = self.bart.decoder(
            inputs_embeds=decoder_input_embeddings,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=input_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        decoder_hidden_states = output.last_hidden_state
        return decoder_hidden_states

    def get_zero_loss(self):
        output = MultiHeadQuantizationOutput()
        output.loss = torch.tensor(0, dtype=torch.float).to(Setting.device)
        return output

    def skip_quant(self, encoder_hidden_states: torch.Tensor):
        input_hidden_states = encoder_hidden_states
        quantized = self.get_zero_loss()
        kl_divergence = torch.tensor(0, dtype=torch.float).to(Setting.device)
        return input_hidden_states, quantized, kl_divergence

    def low_rank(self, encoder_hidden_states: torch.Tensor):
        embeds = self.back_outer_proj(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)
        embeds = self.back_inner_proj(embeds)
        embeds = self.forth_inner_proj(embeds)
        input_hidden_states = self.forth_outer_proj(embeds.transpose(1, 2)).transpose(1, 2)
        quantized = self.get_zero_loss()
        kl_divergence = torch.tensor(0, dtype=torch.float).to(Setting.device)
        return input_hidden_states, quantized, kl_divergence

    def vae(self, encoder_hidden_states: torch.Tensor):
        embeds = self.back_outer_proj(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)
        embeds = self.back_inner_proj(embeds)
        mu, log_var, z = self.vae_encoder(embeds)
        embeds_hat = self.vae_decoder(z)
        embeds_hat = self.forth_inner_proj(embeds_hat)
        input_hidden_states = self.forth_outer_proj(embeds_hat.transpose(1, 2)).transpose(1, 2)
        quantized = self.get_zero_loss()
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return input_hidden_states, quantized, kl_divergence

    def vae_quant(self, encoder_hidden_states: torch.Tensor):
        embeds = self.back_outer_proj(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)
        embeds = self.back_inner_proj(embeds)
        mu, log_var, z = self.vae_encoder(embeds)
        quantized = self.quantizer(z)
        quantized_embeds = torch.stack(quantized.embeds, dim=1)  # [B, H, D]
        embeds_hat = self.vae_decoder(quantized_embeds)
        embeds_hat = self.forth_inner_proj(embeds_hat)
        input_hidden_states = self.forth_outer_proj(embeds_hat.transpose(1, 2)).transpose(1, 2)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return input_hidden_states, quantized, kl_divergence

    def quant(self, encoder_hidden_states: torch.Tensor):
        embeds = self.back_outer_proj(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)
        embeds = self.back_inner_proj(embeds)
        quantized = self.quantizer(embeds)
        quantized_embeds = torch.stack(quantized.embeds, dim=1)  # [B, H, D]
        # quantized_embeds = self.forth_inner_proj(quantized_embeds)
        # input_hidden_states = self.forth_outer_proj(quantized_embeds.transpose(1, 2)).transpose(1, 2)
        kl_divergence = torch.tensor(0, dtype=torch.float).to(Setting.device)
        # return input_hidden_states, quantized, kl_divergence
        return quantized_embeds, quantized, kl_divergence

    def get_codebooks(self):
        # embeds = [qt.codebook.weight for qt in self.quantizer.quantizers]  # H x [C, D]
        # # stack to [H, C, D]
        # embeds = torch.stack(embeds, dim=0)
        # return embeds
        return self.quantizer.get_codebooks()

    def get_codes(self, batch: dict):
        encoder_hidden_states: torch.Tensor = self._encode(batch)

        # noinspection PyArgumentList
        _, quantized, _ = self.handler(encoder_hidden_states)
        indices = torch.stack(quantized.indices, dim=1)  # [B, H]
        item_ids = batch['append'][self.depot.id_col]  # [B]

        return item_ids, indices.squeeze()

    def init(self, batch: dict):
        encoder_hidden_states: torch.Tensor = self._encode(batch)

        embeds = self.back_outer_proj(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)
        embeds = self.back_inner_proj(embeds)
        return embeds

    def forward(self, batch: dict, visualize=False) -> ITOutput:
        encoder_hidden_states: torch.Tensor = self._encode(batch)
        hidden_vector = self.attention(encoder_hidden_states)

        # noinspection PyArgumentList
        input_hidden_states, quantized, kl_divergence = self.handler(hidden_vector)

        # reconstruction loss for encoder_hidden_states and input_hidden_states
        # reconstruction_loss = torch.mean(torch.sum((encoder_hidden_states - input_hidden_states) ** 2, dim=-1))
        reconstruction_loss = torch.tensor(0, dtype=torch.float).to(Setting.device)

        if self.config.kd:
            return ITOutput(
                last_hidden_states=input_hidden_states,
                quantization_loss=quantized.loss,
                generation_loss=torch.tensor(0, dtype=torch.float).to(Setting.device),
                reconstruction_loss=reconstruction_loss,
                kl_divergence=kl_divergence,
                voc_loss=dict(),
            )

        decoder_hidden_states = self._decode(batch, input_hidden_states)

        labels = batch['labels']['labels'].to(Setting.device)  # type: torch.Tensor
        labels_types = batch['labels']['label_types'].to(Setting.device)  # type: torch.Tensor

        generation_loss = torch.tensor(0, dtype=torch.float).to(Setting.device)

        pred_labels = torch.zeros(labels.shape, dtype=torch.long)
        true_labels = torch.zeros(labels.shape, dtype=torch.long)
        voc_loss = dict()
        for voc_name, voc_id in self.inputer.vocab_map.items():
            mask = torch.eq(labels_types, voc_id)
            labels_for_voc = mask * labels
            classifier: BaseClassifier = self.classifier[voc_name]
            output: torch.Tensor = classifier(decoder_hidden_states)
            voc_size = classifier.vocab_size

            distribution = torch.masked_select(
                output, mask.unsqueeze(dim=-1)).view(-1, voc_size).to(Setting.device)
            col_labels = torch.masked_select(labels_for_voc, mask).to(Setting.device)

            loss = self.loss_fct(
                distribution,
                col_labels
            )
            generation_loss += loss
            voc_loss[voc_name] = loss

            if visualize:
                mask = mask.cpu()
                pred_label = torch.argmax(output, dim=-1).cpu()
                # pnt(pred_label[0].tolist())
                # pnt(mask[0].tolist())
                # exit(0)
                pred_label = pred_label.apply_(self.embedding_manager.universal_mapper(voc_name))
                pred_labels[mask] = pred_label[mask]
                true_label = labels_for_voc.cpu().apply_(self.embedding_manager.universal_mapper(voc_name))
                true_labels[mask] = true_label[mask]

        return ITOutput(
            last_hidden_states=decoder_hidden_states,
            pred_labels=pred_labels,
            true_labels=true_labels,
            quantization_loss=quantized.loss,
            generation_loss=generation_loss,
            reconstruction_loss=reconstruction_loss,
            kl_divergence=kl_divergence,
            voc_loss=voc_loss,
        )
