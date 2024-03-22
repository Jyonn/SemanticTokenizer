import argparse
import os

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-k', type=str, default=None)
args = parser.parse_args()

key = args.k

# depth = 3
# code_size = 192

path = f'saving/MIND-small/{key}/export_states/item_embeds.npy'

embeds = np.load(path)

print(embeds.shape)  # [N, D]

# convert numpy to torch

embeds = torch.from_numpy(embeds)

dist = torch.cdist(embeds, embeds)

print(dist.shape)  # [N, N]

# find the nearest neighbor for each item and store the distance, but not self

# nearest_neighbor_dist = torch.min(dist, dim=1).values  # type: torch.Tensor
nearest_neighbor_dist = torch.topk(dist, k=2, dim=1).values[:, 1]  # type: torch.T

# save dist to file, each line is a distance

with open(f'dist.txt', 'w') as f:
    for d in nearest_neighbor_dist:
        f.write(f'{d}\n')




