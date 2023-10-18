import torch
from UniTok import UniDep
from oba import Obj
from pigmento import pnt
from torch import nn

from torch.utils.data import DataLoader

from loader.global_setting import Status
from loader.item_depot import ItemDepot
from model.inputer.concat_inputer import ConcatInputer
from loader.embedding.embedding_manager import EmbeddingManager

from model.it import IT, ITConfig
from loader.base_dataset import BaseDataset

from utils.splitter import Splitter


class Datasets:
    def get_dataset(self, status: str):
        dataset = BaseDataset(
            inputer=self.inputer,
            splitter=self.splitter,
            status=status,
        )
        # dataset.cache()
        return dataset

    def __init__(self, inputer: ConcatInputer):
        self.inputer = inputer

        self.splitter = Splitter()\
            .add(name=Status.TRAIN, weight=8)\
            .add(name=Status.DEV, weight=1)\
            .add(name=Status.TEST, weight=1)

        self.train_set = self.get_dataset(Status.TRAIN)
        self.dev_set = self.get_dataset(Status.DEV)
        self.test_set = self.get_dataset(Status.TEST)

        self._dict = {
            Status.TRAIN: self.train_set,
            Status.DEV: self.dev_set,
            Status.TEST: self.test_set,
        }

    def __getitem__(self, item):
        return self._dict[item]

    def a_set(self):
        return self.train_set or self.dev_set or self.test_set


class ConfigManager:
    def __init__(self, data, embed, model, exp):
        self.data = data
        self.embed = embed
        self.model = model
        self.exp = exp

        pnt('load depots ...')
        self.item_depot = ItemDepot(
            depot=self.data.items.depot,
            order=self.data.items.order,
            append=self.data.items.append,
        )
        pnt('item size: ', len(self.item_depot.depot))
        if self.data.items.union:
            for depot in self.data.items.union:
                self.item_depot.depot.union(UniDep(depot))

        self.it_class = IT
        self.it_config = ITConfig(
            **Obj.raw(self.model.config),
        )

        pnt('build embedding manager ...')
        self.embedding_manager = EmbeddingManager(
            hidden_size=self.it_config.embed_dim,
            same_dim_transform=self.model.config.same_dim_transform,
        )

        pnt('load pretrained embeddings ...')
        for embedding_info in self.embed.embeddings:
            self.embedding_manager.load_pretrained_embedding(**Obj.raw(embedding_info))

        pnt('register embeddings ...')
        self.embedding_manager.register_depot(self.item_depot)
        self.embedding_manager.register_vocab(ConcatInputer.vocab)

        pnt('set <pad> embedding to zeros ...')
        cat_embeddings = self.embedding_manager(ConcatInputer.vocab.name)  # type: nn.Embedding
        cat_embeddings.weight.data[ConcatInputer.PAD] = torch.zeros_like(cat_embeddings.weight.data[ConcatInputer.PAD])

        pnt('build inputer')
        self.inputer = ConcatInputer(
            item_depot=self.item_depot,
            embedding_manager=self.embedding_manager,
            use_sep_token=self.model.config.use_sep_token,
        )

        pnt('build bart model ...')
        self.it = IT(
            config=self.it_config,
            inputer=self.inputer,
            embedding_manager=self.embedding_manager,
        )

        pnt('build datasets ...')
        self.sets = Datasets(inputer=self.inputer)

    def get_loader(self, status):
        return DataLoader(
            dataset=self.sets[status],
            shuffle=status == Status.TRAIN,
            batch_size=self.exp.policy.batch_size,
            pin_memory=self.exp.policy.pin_memory,
            num_workers=5,
        )
