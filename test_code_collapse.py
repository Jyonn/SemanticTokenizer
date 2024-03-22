import argparse
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=int, default=192)
parser.add_argument('-d', type=int, default=3)
parser.add_argument('-k', type=str, default=None)
args = parser.parse_args()

code_size = args.c
depth = args.d
key = args.k

# depth = 3
# code_size = 192

if key:
    path = f'saving/MIND-small/{key}/export/'
else:
    path = f'saving/MIND-small/Depth{depth}-C{code_size}/export/'

code_per_depth = code_size // depth

codes = np.load(os.path.join(path, f'codes.npy'))

print(codes[:20])

# code shape: (65238, 3)
# use set to filter out duplicate codes

codes = set(tuple(c) for c in codes.tolist())

print(len(codes))
