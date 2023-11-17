from collections import OrderedDict
from typing import Optional, Dict, List, Literal, cast

import torch
from UniTok import Vocab

from loader.global_setting import Setting
from model.inputer.base_inputer import BaseInputer


class Pointer:
    def __init__(self):
        self.pos = 0

    def update_input(self, input_ids, value):
        input_ids[self.pos: self.pos + len(value)] = value
        self.pos += len(value)

    def update_special_token(self, input_ids, value):
        value = torch.tensor([value], dtype=torch.long)
        return self.update_input(input_ids, value)


class TypePointer:
    def __init__(self, vocab_map, special_vocab):
        super().__init__()
        self.pos = 0
        self.vocab_map = vocab_map
        self.special_vocab = special_vocab

    def update_input(self, input_ids, input_types, value, vocab_name):
        input_ids[self.pos: self.pos + len(value)] = value
        input_types[self.pos: self.pos + len(value)] = self.vocab_map[vocab_name]
        self.pos += len(value)

    def update_special_token(self, input_ids, input_types, value):
        value = torch.tensor([value], dtype=torch.long)
        return self.update_input(input_ids, input_types, value, self.special_vocab)


class ConcatInputer(BaseInputer):
    vocab = Vocab(name='__cat_inputer_special_ids')
    PAD = vocab.append('[PAD]')
    BOS = vocab.append('[BOS]')
    SEP = vocab.append('[SEP]')
    EOS = vocab.append('[EOS]')

    ENCODER = 'encoder'
    DECODER = 'decoder'

    def __init__(self, use_sep_token, **kwargs):
        super().__init__(**kwargs)

        self.vocab_map = self.embedding_manager.get_vocab_map()
        self.use_sep_token = use_sep_token

        self.max_content_len = self.get_max_content_len()
        self.max_sequence_len = (
                self.max_content_len +  # content
                1 +  # bos
                int(self.use_sep_token) * (len(self.order) - 1) +  # sep
                1  # eos
        )

    def get_max_content_len(self):
        length = 0
        for col in self.order:
            length += self.depot.cols[col].max_length or 1
        return length

    def get_vocabs(self) -> Optional[List[Vocab]]:
        return [self.vocab]

    def get_empty_input(self):
        return torch.ones(self.max_sequence_len, dtype=torch.long) * Setting.UNSET

    def get_autoregressive_labels(self, sample: OrderedDict):
        pointer = Pointer()
        labels = self.get_empty_input()
        pointer_types = Pointer()
        label_types = self.get_empty_input()

        for col in self.order:
            value = sample[col]
            if not isinstance(value, list):
                value = [value]
            value = torch.tensor(value, dtype=torch.long)

            pointer.update_input(labels, value)
            voc_id = self.vocab_map[self.depot.cols[col].voc.name]
            pointer_types.update_input(label_types, torch.ones_like(value) * voc_id)

            if self.use_sep_token and col != self.order[-1]:
                pointer.update_special_token(labels, self.SEP)
                pointer_types.update_special_token(label_types, self.vocab_map[self.vocab.name])

        pointer.update_special_token(labels, self.EOS)
        pointer_types.update_special_token(label_types, self.vocab_map[self.vocab.name])

        return dict(
            labels=labels,
            label_types=label_types,
        )

    def sample_rebuilder(self, sample: OrderedDict, target: Literal['encoder', 'decoder']):
        _pointer = TypePointer(
            vocab_map=self.vocab_map,
            special_vocab=self.vocab.name
        )
        _input_ids = self.get_empty_input()
        _input_types = self.get_empty_input()

        # pointer = Pointer()
        # input_ids = OrderedDict()
        #
        # special_ids = self.get_empty_input()

        if target == self.DECODER:
            # pointer.update_special_token(special_ids, self.BOS)
            _pointer.update_special_token(_input_ids, _input_types, self.BOS)

        for col in self.order:
            value = sample[col]
            if not isinstance(value, list):
                value = [value]
            value = torch.tensor(value, dtype=torch.long)

            # input_id = self.get_empty_input()
            # pointer.update_input(input_id, value)
            # input_ids[col] = input_id
            _pointer.update_input(_input_ids, _input_types, value, self.depot.cols[col].voc.name)

            if self.use_sep_token and col != self.order[-1]:
                # pointer.update_special_token(special_ids, self.SEP)
                _pointer.update_special_token(_input_ids, _input_types, self.SEP)

        if target == self.ENCODER:
            # pointer.update_special_token(special_ids, self.EOS)
            _pointer.update_special_token(_input_ids, _input_types, self.EOS)

        # input_ids[self.vocab.name] = special_ids
        # attention_mask = torch.tensor([1] * pointer.pos + [0] * (self.max_sequence_len - pointer.pos), dtype=torch.long)
        # input_ids[self.vocab.name][pointer.pos:] = self.PAD
        attention_mask = torch.tensor([1] * _pointer.pos + [0] * (self.max_sequence_len - _pointer.pos), dtype=torch.long)

        return dict(
            input_ids=_input_ids,
            input_types=_input_types,
            attention_mask=attention_mask,
        )

    def get_mask(self, batched_samples: Dict[str, torch.Tensor]):
        return batched_samples['attention_mask'].to(Setting.device)

    def get_embeddings(
            self,
            batched_samples: Dict[str, torch.Tensor],
    ):
        input_ids = batched_samples['input_ids'].to(Setting.device)
        input_types = batched_samples['input_types'].to(Setting.device)
        # shape = list(input_ids.values())[0].shape
        shape = input_ids.shape

        input_embeddings = torch.zeros(
            *shape,
            self.embedding_manager.hidden_size,
            dtype=torch.float
        ).to(Setting.device)

        # for col in input_ids:
        #     seq = input_ids[col].to(Setting.device)  # type: torch.Tensor # [B, L]
        #     mask = (seq > Setting.UNSET).long().to(Setting.device)  # type: torch.Tensor  # [B, L]
        #     seq *= mask
        #
        #     embedding = self.embedding_manager(col)(seq)
        #     embedding *= mask.unsqueeze(-1)
        #
        #     input_embeddings += embedding
        #
        # return input_embeddings
        for vocab_name in self.vocab_map:
            vocab_id = self.vocab_map[vocab_name]
            mask = (input_types == vocab_id).long().to(Setting.device)
            seq = input_ids * mask

            embedding = self.embedding_manager(vocab_name, as_vocab=True)(seq)
            embedding *= mask.unsqueeze(-1)

            input_embeddings += embedding

        return input_embeddings