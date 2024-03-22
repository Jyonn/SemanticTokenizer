import os

default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


import argparse

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-k', type=str, default=None)
args = parser.parse_args()

key = args.k

path = f'saving/MIND-small/{key}/export_states/item_embeds.npy'

embeds = np.load(path)


# 使用T-SNE将数据降维到二维
tsne = TSNE(n_components=2, random_state=0)
embeds_2d = tsne.fit_transform(embeds)

# 绘制二维分布图
plt.figure(figsize=(8, 6))
plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1])
plt.xlabel('T-SNE feature 1')
plt.ylabel('T-SNE feature 2')
plt.title('T-SNE Visualization of Embeddings')
plt.savefig('t-sne.png')
plt.close()

