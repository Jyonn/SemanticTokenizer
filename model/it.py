import torch
from torch import nn
from transformers import BartModel
from transformers.modeling_outputs import BaseModelOutput

from loader.embedding.embedding_manager import EmbeddingManager
from model.inputer.concat_inputer import ConcatInputer
from model.quantization.multihead import MultiHeadQuantization
from model.utils.base_classifier import BaseClassifier


class ITOutput:
    def __init__(
            self,
            last_hidden_states: torch.Tensor,
            quantization_loss: torch.Tensor,
            generation_loss: torch.Tensor,
    ):
        self.last_hidden_states = last_hidden_states
        self.quantization_loss = quantization_loss
        self.generation_loss = generation_loss


class ITConfig:
    def __init__(
            self,
            embed_dim: int = 768,
            num_heads: int = 3,
            num_codes: int = 512,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_codes = num_codes


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
        self.bart = BartModel.from_pretrained('facebook/bart-base')  # type: BartModel

        self.back_proj = nn.Linear(self.max_sequence_len, self.config.num_heads)
        self.forth_proj = nn.Linear(self.config.num_heads, self.max_sequence_len)

        self.quantizer = MultiHeadQuantization(
            dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_codes=self.config.num_codes,
        )

        self.classifier = self.embedding_manager.get_classifier()

    def forward(self, batch: dict):
        encoder_attention_mask = self.inputer.get_mask(batch['encoder'])
        encoder_input_embeddings = self.inputer.get_embeddings(batch['encoder'])
        decoder_attention_mask = self.inputer.get_mask(batch['decoder'])
        decoder_input_embeddings = self.inputer.get_embeddings(batch['decoder'])

        output: BaseModelOutput = self.bart.encoder(
            inputs_embeds=encoder_input_embeddings,
            attention_mask=encoder_attention_mask,
            return_dict=True
        )
        encoder_hidden_states = output.last_hidden_state  # [B, L, D]

        # [B, L, D] -> [B, H, D]
        embeds = self.back_proj(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)
        quantized = self.quantizer(embeds)
        quantized_embeds = torch.stack(quantized.embeds, dim=1)  # [B, H, D]
        input_hidden_states = self.forth_proj(quantized_embeds.transpose(1, 2)).transpose(1, 2)

        output: BaseModelOutput = self.bart.decoder(
            inputs_embeds=decoder_input_embeddings,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=input_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        decoder_hidden_states = output.last_hidden_state

        labels = batch['labels']['labels'].to(self.device)  # type: torch.Tensor
        labels_voc = batch['labels']['label_voc'].to(self.device)  # type: torch.Tensor

        generation_loss = torch.tensor(0, dtype=torch.float).to(self.device)
        for voc_name, voc_id in self.inputer.vocab_map.items():
            voc_mask = torch.eq(labels_voc, voc_id)
            labels_for_voc = voc_mask * labels
            classifier: BaseClassifier = self.classifier[voc_name]
            output: torch.Tensor = classifier(decoder_hidden_states)
            voc_size = classifier.vocab_size

            distribution = torch.masked_select(
                output, voc_mask.unsqueeze(dim=-1)).view(-1, voc_size).to(self.device)
            col_labels = torch.masked_select(labels_for_voc, voc_mask).to(self.device)

            loss = self.loss_fct(
                distribution,
                col_labels
            )
            generation_loss += loss

        return ITOutput(
            last_hidden_states=decoder_hidden_states,
            quantization_loss=quantized.loss,
            generation_loss=generation_loss,
        )
