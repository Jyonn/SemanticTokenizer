import random

import torch
from tqdm import tqdm

from loader.global_setting import Setting

from loader.status import Status
from model.it import IT
from model.utils.nr_depot import ItemDepot
from loader.base_dataset import BaseDataset
from utils.stacker import Stacker
from utils.timer import Timer


class DataManager:
    def __init__(
            self,
            item_nrd: ItemDepot,
            it: IT,
    ):
        self.status = Status()

        self.timer = Timer(activate=True)

        self.stacker = Stacker(aggregator=torch.stack)
        self.item_dataset = BaseDataset(item_depot=item_nrd)
        self.item_inputer = it.inputer

    def rebuild_sample(self, sample):
        len_clicks = len(sample[self.clicks_col])
        sample[self.clicks_mask_col] = [1] * len_clicks + [0] * (self.max_click_num - len_clicks)
        if self.use_news_content:
            sample[self.clicks_col].extend([0] * (self.max_click_num - len_clicks))
        if not isinstance(sample[self.candidate_col], list):
            sample[self.candidate_col] = [sample[self.candidate_col]]

        if self.use_news_content and not self.recommender.llm_skip and not self.recommender.fast_doc_eval:
            sample[self.clicks_col] = self.stacker([self.doc_cache[nid] for nid in sample[self.clicks_col]])
        else:
            sample[self.candidate_col] = torch.tensor(sample[self.candidate_col], dtype=torch.long)

        return sample
