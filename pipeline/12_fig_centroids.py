from pipeline.paths import METADATA, RESULTS, TABLES

"""
@author: Claudio

Tables and results used in the article

"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# PATH
path_demo = METADATA
path_tables = TABLES
path_results = RESULTS


from openTSNE import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

#df = pd.read_pickle(path_results+'df_centroids_sents.pkl')
df = pd.read_pickle(path_results+'df_centroids_sents_mono.pkl')


# REDUCTION OF DIMENSIONS AND VISUALIZATION

centroid_array = df['centroid'].to_numpy()
centroid_array = np.vstack(centroid_array)
tsne = TSNE(n_components=2, perplexity=45, initialization="pca", metric="euclidean", n_jobs=-1, random_state=42)
tsne_embeddings = tsne.fit(centroid_array)
df['tsne_embedding_1'] = tsne_embeddings[:, 0]
df['tsne_embedding_2'] = tsne_embeddings[:, 1]

# CLASSIFY PICTURES

# KNN
knn = KNeighborsClassifier(4)
knn.fit(tsne_embeddings, df['bild'])  # Assuming true labels are available
cluster_labels = knn.predict(tsne_embeddings)
conf_matrix = confusion_matrix(df['bild'], cluster_labels)

# SCORES
TP = np.diag(conf_matrix)
FP = np.sum(conf_matrix, axis=0) - TP
FN = np.sum(conf_matrix, axis=1) - TP
TN = np.sum(conf_matrix) - (TP + FP + FN)

# Calculate precision, recall, accuracy, F1-score, specificity, FPR, and FNR
precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = np.sum(TP) / np.sum(conf_matrix)
f1_score = 2 * (precision * recall) / (precision + recall)
specificity = TN / (TN + FP)
fpr = FP / (TN + FP)
fnr = FN / (TP + FN)
avg_specificity = np.mean(specificity)
avg_fpr = np.mean(fpr)
avg_fnr = np.mean(fnr)

# Print the scores
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1-score:", f1_score)
print("Average Specificity:", avg_specificity)
print("Average False Positive Rate:", avg_fpr)
print("Average False Negative Rate:", avg_fnr)


# CLASSIFY GROUPS

# KNN
knn = KNeighborsClassifier(4)
knn.fit(tsne_embeddings, df['group'])  # Assuming true labels are available
cluster_labels = knn.predict(tsne_embeddings)
labels = df['group'].unique()
conf_matrix = confusion_matrix(df['group'], cluster_labels, labels=labels)

# SCORES
TP = np.diag(conf_matrix)
FP = np.sum(conf_matrix, axis=0) - TP
FN = np.sum(conf_matrix, axis=1) - TP
TN = np.sum(conf_matrix) - (TP + FP + FN)

# Calculate precision, recall, accuracy, F1-score, specificity, FPR, and FNR
precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = np.sum(TP) / np.sum(conf_matrix)
f1_score = 2 * (precision * recall) / (precision + recall)
specificity = TN / (TN + FP)
fpr = FP / (TN + FP)
fnr = FN / (TP + FN)
avg_specificity = np.mean(specificity)
avg_fpr = np.mean(fpr)
avg_fnr = np.mean(fnr)

# Print the scores
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1-score:", f1_score)
print("Average Specificity:", avg_specificity)
print("Average False Positive Rate:", avg_fpr)
print("Average False Negative Rate:", avg_fnr)


# SCATTER PLOTS PICTURE
sns.set_palette("colorblind")
sns.scatterplot(x=df['tsne_embedding_1'], y=df['tsne_embedding_2'], hue=df['bild'], s=16)
plt.xlabel('X')
plt.ylabel('Y')
#plt.title('Speech centroids from sentence embeddings in 2D - Picture classification')
plt.legend(title='Picture', loc='upper right')
#plt.show()
plt.savefig(path_results+'tsne_picture_clusters_mono.jpg', dpi=500)

# SCATTER PLOTS GROUPS
plt.figure()
hue_order = ['HC', 'MDD', 'SSD']
sns.set_palette("colorblind")
sns.scatterplot(x=df['tsne_embedding_1'], y=df['tsne_embedding_2'], hue_order=hue_order, hue=df['group'], s=16)
plt.xlabel('X')
plt.ylabel('Y')
#plt.title('Speech centroids from sentence embeddings in 2D - Group classification')
plt.legend(title='Group', loc='upper right')
#plt.show()
plt.savefig(path_results+'tsne_group_clusters_mono.jpg', dpi=500)

