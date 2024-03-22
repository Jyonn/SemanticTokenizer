import os

import numpy as np
import pandas as pd
from UniTok import UniDep, Vocab, Fut

dataset = 'MIND-small-wo-abs'
target = 'user' if dataset.endswith('user') else 'news-wo-abs'
key = 'uid' if target == 'user' else 'nid'
folder = 'user-grp' if target == 'user' else 'news'

data = np.load(f'{dataset}.rq.npy', allow_pickle=True).item()
indices = data['indices']
codes = data['codes']

indices = [index.tolist() for index in indices]

for depth in range(len(indices)):
    print(depth + 1)

    items = set()
    for i in range(len(indices[0])):
        item = []
        for index in indices[:depth + 1]:
            item.append(index[i])
        items.add(tuple(item))

    print(len(items))

sizes = [64] * 10

depth = 6
items = []
total_size = sum(sizes[:depth])
print(f'total size: {total_size}')
for i in range(len(indices[0])):
    item = []
    offset = 0
    code_size = 64
    for index in indices[:depth]:
        item.append(index[i] + offset)
        offset += code_size
        # code_size *= 2
    items.append(item)

    # print(len(items))

codes = np.concatenate(codes[:depth], axis=0)
np.save(f'{target}.manual-kmeans-d{depth}.codebooks.npy', codes)
print(codes.shape)


if not os.path.exists(f'data/MIND-small/{target}-code'):
    depot = UniDep(f'data/MIND-small/{folder}')
    vocab_size = depot.cols[key].voc.size
    data = {
        key: list(range(vocab_size)),
    }
    df = pd.DataFrame(data)
    Fut(
        df,
        depot,
        id_col=key,
    ).store(f'data/MIND-small/{target}-code')


depot = UniDep(f'data/MIND-small/{target}-code')
depot.set_col(
    name=f'{target}-manual-kmeans-d{depth}',
    values=items,
    vocab=Vocab(name=f'{target}-manual-kmeans-d{depth}').reserve(total_size),
)
depot.export(f'data/MIND-small/{target}-code')
