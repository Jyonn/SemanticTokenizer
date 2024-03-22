import os

import torch
from torch import nn
from transformers import BartModel, BartConfig
from transformers.modeling_outputs import BaseModelOutput

from loader.embedding.embedding_manager import EmbeddingManager
from loader.global_setting import Setting
from model.handlers.base_handler import BaseHandler, HandlerOutput
from model.handlers.quant_handler import QuantHandler
from model.handlers.vae_handler import VAEHandler
from model.handlers.vae_quant_handler import VAEQuantHandler
from model.inputer.concat_inputer import ConcatInputer
from model.quantization.residual import ResidualQuantization
# from model.quantization.residual_v2 import ResidualQuantizationV2
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
            **kwargs,
    ):
        self.last_hidden_states = last_hidden_states
        self.pred_labels = pred_labels
        self.true_labels = true_labels

        self.quantization_loss = quantization_loss
        self.generation_loss = generation_loss
        self.reconstruction_loss = reconstruction_loss
        self.kl_divergence = kl_divergence
        self.voc_loss = voc_loss

        for k, v in kwargs.items():
            setattr(self, k, v)


class Handler:
    BASELINE = 'baseline'
    LOW_RANK = 'low_rank'
    VAE = 'vae'
    VAE_QUANT = 'vae_quant'
    QUANT = 'quant'


class ITConfig:
    def __init__(
            self,
            embed_dim: int = 768,
            hidden_size: int = 256,
            num_heads: int = 3,
            num_codes: int = 512,
            residual_depth: int = 4,
            commitment_cost: float = 0.0,
            handler: str = Handler.BASELINE,
            use_pretrained_model: bool = True,
            num_layers: int = 6,
            num_attention_heads: int = 12,
            # baseline: bool = False,
            # low_rank: bool = False,
            # vae: bool = False,
            # vae_quant: bool = False,
            # kd: bool = False,
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
        self.handler = handler
        self.use_pretrained_model = use_pretrained_model
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        # self.baseline = baseline
        # self.low_rank = low_rank
        # self.vae = vae
        # self.vae_quant = vae_quant
        # self.kd = kd





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

        if self.config.use_pretrained_model:
            bart_path = '/data1/qijiong/Code/bart-base'
            if not os.path.exists(bart_path):
                bart_path = 'facebook/bart-base'
            self.bart = BartModel.from_pretrained(bart_path)  # type: BartModel
        else:
            self.bart = BartModel(
                config=BartConfig(
                    vocab_size=5,
                    d_model=self.config.embed_dim,
                    encoder_layers=self.config.num_layers,
                    decoder_layers=self.config.num_layers,
                    encoder_attention_heads=self.config.num_attention_heads,
                    decoder_attention_heads=self.config.num_attention_heads,
                    encoder_ffn_dim=self.config.embed_dim * 4,
                    decoder_ffn_dim=self.config.embed_dim * 4,
                )
            )

        # self.back_outer_proj = nn.Linear(self.max_sequence_len, self.config.num_heads)
        # self.back_inner_proj = nn.Linear(self.config.embed_dim, self.config.hidden_size)
        # self.back_inner_proj = lambda x: x
        # self.vae_encoder = VAEEncoder(self.config.embed_dim)

        # self.forth_inner_proj = nn.Linear(self.config.hidden_size, self.config.embed_dim)
        # self.forth_inner_proj = lambda x: x
        # self.forth_outer_proj = nn.Linear(self.config.num_heads, self.max_sequence_len)
        # self.vae_decoder = VAEDecoder(self.config.embed_dim)

        self.attention = AdditiveAttention(self.config.embed_dim, self.config.hidden_size)

        # self.use_dnn = self.config.dnn_dim is not None
        # self.quant_dim = self.config.dnn_dim or self.config.embed_dim
        # if self.use_dnn:
        #     self.dnn_1 = nn.Linear(self.config.embed_dim, self.config.dnn_dim)
        #     self.dnn_2 = nn.Linear(self.config.dnn_dim, self.config.embed_dim)
            # self.full_codes = nn.Embedding(
            #     self.config.num_codes,
            #     self.config.embed_dim,
            # )

        # if not self.config.baseline:
        #     self.quantizer = MultiHeadQuantizationV2(
        #         feature_size=self.config.embed_dim,
        #         num_heads=self.config.num_heads,
        #         num_codes=self.config.num_codes,
        #         beta=self.config.commitment_cost,
        #         kmeans_init=True,
        #         affine_lr=10.0,
        #         replace_freq=10,
        #         sync_nu=0.2,
        #         dim=-1,
        #     )

        # self.quantizer = ResidualQuantizationV2(
        #     feature_size=self.config.embed_dim,
        #     num_codes=self.config.num_codes,
        #     depth=self.config.residual_depth,
        #     beta=self.config.commitment_cost,
        #     kmeans_init=True,
        #     affine_lr=10.0,
        #     replace_freq=10,
        #     sync_nu=0.2,
        #     dim=-1,
        # )
        self.quantizer = ResidualQuantization(
            num_layers=self.config.residual_depth,
            num_codes=self.config.num_codes // self.config.residual_depth,
            commitment_cost=self.config.commitment_cost,
            threshold_disappearance=256,
            dim=self.config.embed_dim,
        )

        self.is_residual_vq = not self.config.handler == Handler.BASELINE and isinstance(self.quantizer, ResidualQuantization)
        self.classifier = self.embedding_manager.get_classifier()
        # self.classifier = self.embedding_manager.get_universal_classifier()
        self.loss_fct = nn.CrossEntropyLoss()

        # if self.config.kd:
        #     # frozen bart and classifier
        #     for param in self.bart.parameters():
        #         param.requires_grad = False
        #     for param in self.classifier.parameters():
        #         param.requires_grad = False
        #     for param in self.attention.parameters():
        #         param.requires_grad = False

        self.handler = self.get_handler()

        Setting.it = self

    def get_handler(self):
        if self.config.handler == Handler.BASELINE:
            return BaseHandler()
        if self.config.handler == Handler.QUANT:
            return QuantHandler()
        if self.config.handler == Handler.VAE:
            return VAEHandler()
        if self.config.handler == Handler.VAE_QUANT:
            return VAEQuantHandler()

        raise ValueError(f'handler {self.config.handler} not supported')

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

    def _decode(self, batch, states):
        encoder_attention_mask = self.inputer.get_mask(batch['encoder'])
        decoder_attention_mask = self.inputer.get_mask(batch['decoder'])
        decoder_input_embeddings = self.inputer.get_embeddings(batch['decoder'])

        # if self.is_residual_vq:
        if states.dim() == 2:
            seq_len = encoder_attention_mask.shape[1]
            states = states.unsqueeze(dim=1).repeat(1, seq_len, 1)
        encoder_attention_mask = torch.zeros_like(encoder_attention_mask)
        encoder_attention_mask[:, 0] = 1

        output: BaseModelOutput = self.bart.decoder(
            inputs_embeds=decoder_input_embeddings,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        decoder_hidden_states = output.last_hidden_state
        return decoder_hidden_states

    def get_codebooks(self):
        return self.quantizer.get_codebooks()

    def get_embeds(self, batch: dict):
        encoder_hidden_states: torch.Tensor = self._encode(batch)
        item_ids = batch['append'][self.depot.id_col]  # [B]
        embeds = self.attention(encoder_hidden_states)
        return item_ids, embeds

    def get_codes(self, batch: dict):
        encoder_hidden_states: torch.Tensor = self._encode(batch)

        # noinspection PyArgumentList
        _, quantized, _, _ = self.handler(encoder_hidden_states)
        # pnt(quantized.indices.shape)
        indices = quantized.indices
        if isinstance(indices, list):
            indices = torch.stack(quantized.indices, dim=1)  # [B, H]
        item_ids = batch['append'][self.depot.id_col]  # [B]

        return item_ids, indices.squeeze()

    def init(self, batch: dict):
        encoder_hidden_states: torch.Tensor = self._encode(batch)
        embeds = self.attention(encoder_hidden_states)
        # if self.use_dnn:
        #     embeds = self.dnn_1(embeds)
        return embeds

    def generate(self, codes):
        codebooks = self.get_codebooks()
        code_embeds = codebooks[codes]

        # auto-regressive decoding



        # encoder_attention_mask = self.inputer.get_mask(batch['encoder'])
        # decoder_attention_mask = self.inputer.get_mask(batch['decoder'])
        # decoder_input_embeddings = self.inputer.get_embeddings(batch['decoder'])
        #
        # if self.is_residual_vq:
        #     seq_len = encoder_attention_mask.shape[1]
        #     input_hidden_states = input_hidden_states.unsqueeze(dim=1).repeat(1, seq_len, 1)
        #     encoder_attention_mask = torch.zeros_like(encoder_attention_mask)
        #     encoder_attention_mask[:, 0] = 1
        #
        # output: BaseModelOutput = self.bart.decoder(
        #     inputs_embeds=decoder_input_embeddings,
        #     attention_mask=decoder_attention_mask,
        #     encoder_hidden_states=input_hidden_states,
        #     encoder_attention_mask=encoder_attention_mask,
        #     return_dict=True,
        # )
        # decoder_hidden_states = output.last_hidden_state
        # return decoder_hidden_states

    def forward(self, batch: dict, visualize=False) -> ITOutput:
        encoder_hidden_states: torch.Tensor = self._encode(batch)

        # noinspection PyArgumentList
        handler_output: HandlerOutput = self.handler(encoder_hidden_states)

        decoder_hidden_states = self._decode(batch, handler_output.states)

        labels = batch['encoder']['input_ids'].to(Setting.device)  # type: torch.Tensor
        labels_types = batch['encoder']['input_types'].to(Setting.device)  # type: torch.Tensor

        generation_loss = torch.tensor(0, dtype=torch.float).to(Setting.device)

        pred_labels = torch.zeros(labels.shape, dtype=torch.long)
        true_labels = torch.zeros(labels.shape, dtype=torch.long)

        # mask = torch.not_equal(labels, Setting.UNSET)
        # labels = labels * mask
        # output = self.classifier(decoder_hidden_states)
        # voc_size = self.classifier.vocab_size
        #
        # distribution = torch.masked_select(
        #     output, mask.unsqueeze(dim=-1)).view(-1, voc_size).to(Setting.device)
        # col_labels = torch.masked_select(labels, mask).to(Setting.device)
        #
        # if not torch.sum(col_labels):
        #     loss = torch.tensor(0, dtype=torch.float).to(Setting.device)
        # else:
        #     loss = self.loss_fct(
        #         distribution,
        #         col_labels
        #     )
        #
        # generation_loss += loss
        # voc_loss = dict(universal=loss)
        #
        # if visualize:
        #     mask = mask.cpu()
        #     pred_label = torch.argmax(output, dim=-1).cpu()
        #     pred_labels[mask] = pred_label[mask]
        #     true_label = labels.cpu()
        #     true_labels[mask] = true_label[mask]

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

            if not torch.sum(col_labels):
                loss = torch.tensor(0, dtype=torch.float).to(Setting.device)
            else:
                loss = self.loss_fct(
                    distribution,
                    col_labels
                )
            generation_loss += loss
            voc_loss[voc_name] = loss

            if visualize:
                mask = mask.cpu()
                pred_label = torch.argmax(output, dim=-1).cpu()
                pred_label = pred_label.apply_(self.embedding_manager.universal_mapper(voc_name))
                pred_labels[mask] = pred_label[mask]
                true_label = labels_for_voc.cpu().apply_(self.embedding_manager.universal_mapper(voc_name))
                true_labels[mask] = true_label[mask]

        return ITOutput(
            last_hidden_states=decoder_hidden_states,
            pred_labels=pred_labels,
            true_labels=true_labels,
            quantization_loss=handler_output.quantized.loss,
            generation_loss=generation_loss,
            reconstruction_loss=handler_output.recon,
            kl_divergence=handler_output.kl,
            voc_loss=voc_loss,
            # states=handler_output.states.sum(dim=-1),
            # indices=handler_output.quantized.indices,
        )
