import os

default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import argparse

import numpy as np
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser()
parser.add_argument('-k', type=str, default=None)
args = parser.parse_args()

key = args.k

depth = 0
num_codes = 64

dataset = 'MIND-small-wo-abs'

path = f'saving/{dataset}/{key}/export_states/item_embeds.npy'

embeds = np.load(path)


# residual quantization, iteratively K-means

indices = []
codes = []


while True:
    depth += 1
    print(f'Quantizing depth {depth}')
    kmeans = KMeans(n_clusters=num_codes, random_state=0, verbose=0, max_iter=1000).fit(embeds)
    # embeds = kmeans.cluster_centers_[kmeans.labels_]
    labels = kmeans.labels_  # type: np.ndarray  # [N]
    centers = kmeans.cluster_centers_  # type: np.ndarray  # [N, D]
    indices.append(labels)
    codes.append(centers)

    print(labels.shape)
    print(centers.shape)
    embeds -= centers[labels]
    # num_codes *= 2

    code_map = dict()
    for i in range(len(labels)):
        _code = []
        for j in range(depth):
            _code.append(indices[j][i])
        _code = tuple(_code)
        if _code not in code_map:
            code_map[_code] = []
        code_map[_code].append(i)

    count = 0
    for _code in code_map:
        if len(code_map[_code]) == 1:
            count += 1

    print(f'unique codes: {count}')

    if depth == 10 or count == len(labels):
        break


# concat indices and codes
# indices: [depth x N]
# codes: [[N, D], [N*2, D], [N*4, D], ...

# indices = np.concatenate(indices, axis=0)
# codes = np.concatenate(codes, axis=0)
# codes =

# print(indices.shape)
# print(codes.shape)

# save indices and codes

# np.save(f'indices.npy', indices)
# np.save(f'codes.npy', codes)
data = dict(
        indices=indices,
        codes=codes,
)

for index in indices:
    print(index.shape)

for code in codes:
    print(code.shape)

np.save(f'{dataset}.rq.npy', data, allow_pickle=True)
