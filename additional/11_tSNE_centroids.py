# Additional analysis; check additional/README.md before running.
from pipeline.paths import TEXTS, METADATA, ADDITIONAL_RESULTS

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:45:25 2024

@author: Claudio
"""

import numpy as np
import pandas as pd
from openTSNE import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# PATH INPUT
path_input = TEXTS
path_demo = METADATA
path_results = ADDITIONAL_RESULTS


""" UMAP - REDUCTION OF DIMENSIONALITY OF CENTROIDS SENTENCES """

# df_cent = pd.read_pickle(path_results+'df_centroids_sents.pkl')
df_cent = pd.read_pickle(path_results+'df_centroids_words.pkl')

centroid_array = df_cent['centroid'].to_numpy()
centroid_array = np.vstack(centroid_array)

tsne = TSNE(n_components=2, perplexity=45, n_jobs=-1, random_state=42)
tsne_embeddings = tsne.fit(centroid_array)
df_cent['tsne_embedding_1'] = tsne_embeddings[:, 0]
df_cent['tsne_embedding_2'] = tsne_embeddings[:, 1]


# SCATTER PLOTS
sns.set_palette("colorblind")
sns.scatterplot(x=df_cent['tsne_embedding_1'], y=df_cent['tsne_embedding_2'], hue=df_cent['bild'])
plt.xlabel('t-SNE Embedding 1')
plt.ylabel('t-SNE Embedding 2')
plt.title('t-SNE Embedding with Picture Colors')
plt.legend(title='Picture', loc='upper right')
plt.show()

sns.set_palette("colorblind")
sns.scatterplot(x=df_cent['tsne_embedding_1'], y=df_cent['tsne_embedding_2'], hue=df_cent['group'])
plt.xlabel('t-SNE Embedding 1')
plt.ylabel('t-SNE Embedding 2')
plt.title('t-SNE Embedding with Group Colors')
plt.legend(title='Group', loc='upper right')
plt.show()
