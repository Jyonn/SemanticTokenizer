from collections import OrderedDict

from torch.utils.data import Dataset
from tqdm import tqdm

from model.inputer.concat_inputer import ConcatInputer
from loader.item_depot import ItemDepot
from utils.splitter import Splitter


class BaseDataset(Dataset):
    def __init__(
            self,
            inputer: ConcatInputer,
            splitter: Splitter,
            status: str,
    ):
        self.item_depot = inputer.item_depot
        self.inputer = inputer

        self.depot = self.item_depot.depot
        self.order = self.item_depot.order
        self.append = self.item_depot.append
        self.append_checker()

        self.sample_size = self.depot.sample_size

        self.splitter = splitter
        self.status = status
        # self.split_range = (0, self.sample_size)
        self.split_range = splitter.divide(self.sample_size)[self.status]

        self._cached = False
        self._cache = []

    def append_checker(self):
        for col in self.append:
            if self.depot.is_list_col(col):
                self.print(f'{col} is a list col, please do list align in task carefully')

    def __getitem__(self, index):
        if self._cached:
            return self._cache[index]
        index += self.split_range[0]
        return self.pack_sample(index)

    def __len__(self):
        mode_range = self.split_range
        return mode_range[1] - mode_range[0]

    def pack_sample(self, index):
        sample = self.depot[index]
        order = OrderedDict()
        for col in self.order:
            order[col] = sample[col]
        append = OrderedDict()
        for col in self.append:
            append[col] = sample[col]
        encoder = self.inputer.sample_rebuilder(order, self.inputer.ENCODER)
        decoder = self.inputer.sample_rebuilder(order, self.inputer.DECODER)
        labels = self.inputer.get_autoregressive_labels(order)
        return dict(
            encoder=encoder,
            decoder=decoder,
            labels=labels,
            append=append,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def cache(self):
        self._cached = False
        self._cache = []
        for i in tqdm(range(len(self))):
            self._cache.append(self[i])
        self._cached = True
