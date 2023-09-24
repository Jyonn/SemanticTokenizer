import os

import pandas as pd
from UniTok import Vocab, UniTok, Column
from UniTok.tok import IdTok, EntTok, BaseTok
from transformers import BartTokenizer


class BartTok(BaseTok):
    """
        Bart Tokenizer

        Args:
            name: name of the tokenizer
            model_name: model name in huggingface
        """
    return_list = True

    def __init__(self, name, model_name='facebook/bart-base'):
        super(BartTok, self).__init__(name=name)
        self.tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        vocab_map = self.tokenizer.get_vocab()
        vocabs = []
        start_index = 0
        for key in vocab_map:
            assert vocab_map[key] == start_index
            start_index += 1
            vocabs.append(key)
        self.vocab.extend(vocabs)

    def t(self, obj) -> [int, list]:
        if pd.notnull(obj):
            ts = self.tokenizer.tokenize(obj)
            ids = self.tokenizer.convert_tokens_to_ids(ts)
        else:
            ids = []
        return ids


class Processor:
    def __init__(self, data_dir, store_dir):
        self.data_dir = data_dir
        self.store_dir = store_dir

        os.makedirs(self.store_dir, exist_ok=True)

        self.nid = Vocab(name='nid')
        self.uid = Vocab(name='uid')

    def read_news_data(self, mode):
        return pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, mode, 'news.tsv'),
            sep='\t',
            names=['nid', 'cat', 'subcat', 'title', 'abs', 'url', 'tit_ent', 'abs_ent'],
            usecols=['nid', 'cat', 'subcat', 'title', 'abs'],
        )

    def get_news_tok(self, max_title_len=0, max_abs_len=0):
        txt_tok = BartTok(name='bart', model_name='facebook/bart-base')

        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.nid)
        )).add_col(
            col='cat',
            tok=EntTok,
        ).add_col(
            col='subcat',
            tok=EntTok,
        ).add_col(Column(
            name='title',
            tok=txt_tok,
            max_length=max_title_len,
        )).add_col(Column(
            name='abs',
            tok=txt_tok,
            max_length=max_abs_len,
        ))

    def combine_news_data(self):
        news_train_df = self.read_news_data('train')
        news_dev_df = self.read_news_data('dev')
        news_df = pd.concat([news_train_df, news_dev_df])
        news_df = news_df.drop_duplicates(['nid'])
        return news_df

    def analyse_news(self):
        tok = self.get_news_tok(
            max_title_len=0,
            max_abs_len=0
        )
        df = self.combine_news_data()
        tok.read(df).analyse()

    def tokenize(self):
        news_tok = self.get_news_tok(
            max_title_len=20,
            max_abs_len=50
        )
        news_df = self.combine_news_data()
        news_tok.read_file(news_df).tokenize().store_data(os.path.join(self.store_dir, 'news'))


if __name__ == '__main__':
    p = Processor(
        data_dir='/data1/qijiong/Data/MIND/',
        store_dir='../../data/MIND-small',
    )
    p.tokenize()
