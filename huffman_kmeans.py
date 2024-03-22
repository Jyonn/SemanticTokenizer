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

num_codes = 64

dataset = 'MIND-small-wo-abs'

path = f'saving/{dataset}/{key}/export_states/item_embeds.npy'

_embeds = np.load(path)


# residual quantization, iteratively K-means

num_items = _embeds.shape[0]
mixed_indices = list(range(num_items))

indices = []
codes = []


depth = 0

while True:
    depth += 1
    print(f'Quantizing depth {depth}')
    print(f'current item size: {len(mixed_indices)}')
    embeds = _embeds[mixed_indices]
    kmeans = KMeans(n_clusters=num_codes, random_state=0, verbose=0, max_iter=1000).fit(embeds)
    # embeds = kmeans.cluster_centers_[kmeans.labels_]
    _labels = kmeans.labels_  # type: np.ndarray  # [N]
    labels = [-1] * num_items
    for ci, i in enumerate(mixed_indices):
        labels[i] = _labels[ci]
    centers = kmeans.cluster_centers_  # type: np.ndarray  # [N, D]
    indices.append(np.array(labels))
    codes.append(centers)

    embeds -= centers[_labels]
    _embeds[mixed_indices] = embeds
    # num_codes *= 2

    code_map = dict()
    for i in mixed_indices:
        _code = []
        for j in range(depth):
            _code.append(indices[j][i])
        _code = tuple(_code)
        if _code not in code_map:
            code_map[_code] = []
        code_map[_code].append(i)

    mixed_indices = []
    for _code in code_map:
        if len(code_map[_code]) > 1:
            mixed_indices.extend(code_map[_code])
    if len(mixed_indices) < num_codes or depth == 10:
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

np.save(f'{dataset}.huffman.npy', data, allow_pickle=True)
