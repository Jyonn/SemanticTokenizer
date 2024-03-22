import os.path

import numpy as np
import pandas as pd
from UniTok import UniDep, Fut, Vocab

depth = 3
code_size = 192
path = f'saving/MIND-small/Depth{depth}-C{code_size}/export/'

key = f'd{depth}-c{code_size}'

code_per_depth = code_size // depth


codebooks = np.load(os.path.join(path, f'codebooks.npy'))
codes = np.load(os.path.join(path, f'codes.npy'))

# codebooks = codebooks.reshape(depth, code_per_depth, codebooks.shape[-1])
#
# for i in range(depth):
#     codes[:, i] -= i * code_per_depth

np.save(os.path.join(path, f'{key}.codebooks.npy'), codebooks)
np.save(os.path.join(path, f'{key}.codes.npy'), codes)

if not os.path.exists('data/MIND-small/news-code'):
    depot = UniDep('data/MIND-small/news')
    data = {
        'nid': list(range(codes.shape[0])),
    }
    df = pd.DataFrame(data)
    Fut(
        df,
        depot,
        id_col='nid',
    ).store('data/MIND-small/news-code')

depot = UniDep('data/MIND-small/news-code')
depot.set_col(
    name=key,
    values=codes.tolist(),
    vocab=Vocab(name=key).reserve(code_size),
)
depot.export('data/MIND-small/news-code')
