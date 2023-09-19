from torch import nn
from transformers import BartModel
from transformers.modeling_outputs import BaseModelOutput

from loader.embedding.embedding_manager import EmbeddingManager
from model.inputer.concat_inputer import ConcatInputer


class ITConfig:
    def __init__(
            self,
            num_codes: int = 3,
    ):
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
        self.max_sequence_len = inputer.max_sequence_len

        self.embedding_manager = embedding_manager
        self.bart_model = BartModel.from_pretrained('facebook/bart-base')  # type: BartModel

        self.back_proj = nn.Linear(self.max_sequence_len, self.config.num_codes)
        self.forth_proj = nn.Linear(self.config.num_codes, self.max_sequence_len)

    def forward(self, batch: dict):
        attention_mask = self.inputer.get_mask(batch)
        input_embeddings = self.inputer.get_embeddings(batch)

        output: BaseModelOutput = self.bart_model.encoder(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = output.last_hidden_state  # [B, L, D]

        # [B, L, D] -> [B, K, D]
        codes = self.back_proj(hidden_states.transpose(1, 2)).transpose(1, 2)





