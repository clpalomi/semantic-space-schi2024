from pipeline.text_measures import read_files_sent, slope_sign_changes

# Additional analysis; check additional/README.md before running.
from pipeline.paths import TEXTS, METADATA, SANTANDER, ADDITIONAL_RESULTS

"""
@author: Claudio
"""

import os
import re
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from openTSNE import TSNE
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# PATHS
path_input = TEXTS
path_demo = METADATA
path_santander = SANTANDER
path_results_german = ADDITIONAL_RESULTS


""" FUNCTIONS"""


# CREATE A DICTIONARY OF TEXT FROM FOLDER


# SEMATIC SIMILARITY BETWEEN SENTENCES (SENTENCE TRANSFORMER BERT)
def sent_sem_sim(sentences):
    # model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') # 768 
    sentence_embeddings = model.encode(sentences)
    all_similarities = []
    for i in range(len(sentences)-1):
        emb1 = sentence_embeddings[i].reshape(1, -1)
        emb2 = sentence_embeddings[i+1].reshape(1, -1)
        similarity_matrix = cosine_similarity(emb1, emb2)
        all_similarities.append(similarity_matrix[0, 0])
    return all_similarities


# DISPERSION W/R CENTROID
def sent_embedding_centroid(sentences):
    # model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') # 768 dimensions
    sentence_embeddings = model.encode(sentences)
    # Calculate centroid
    centroid = np.mean(sentence_embeddings, axis=0)
    # Calculate dispersion
    dispersion = np.mean([cosine_similarity(centroid.reshape(1, -1), emb.reshape(1, -1))[0, 0] for emb in sentence_embeddings])
    # Calculate sum of euclidean distances between consecutive embeddings
    euclidean_distances = [np.linalg.norm(emb1 - emb2) for emb1, emb2 in zip(sentence_embeddings[:-1], sentence_embeddings[1:])]
    sum_distances = np.sum(euclidean_distances)
    return [centroid, dispersion, sum_distances]


# CALCULATE NUMBER OF TIMES CROSSING THE AVERAGE
def crossings_average(sem_list):
    if len(sem_list) == 1:
        return [0, 0]
    # Calculate the average value of the list
    average_value = np.mean(sem_list)
    # Initialize counter for crossings
    crossings_count = 0
    # Iterate through consecutive pairs of values, starting from the second element
    for i in range(1, len(sem_list)):
        current_value = sem_list[i]
        previous_value = sem_list[i - 1]
        # Check if one value is above average and the other is not
        if (current_value > average_value and previous_value <= average_value) or \
           (previous_value > average_value and current_value <= average_value):
            crossings_count += 1
    return [crossings_count, crossings_count/(len(sem_list)-1)]


# SLOPE SIGN CHANGES




""" READ GERMAN DATA: SPLIT SPEECH INTO SENTENCES """

# SAVE EACH LIST OF WORDS
text_german = read_files_sent(path_input)
text_german = {key[:-4]: value for key, value in text_german.items()}
# IMPORT DEMOGRAPHICS
groups = pd.read_excel(path_demo+"groups.xlsx")
# UNIQUE NAMES
names_german = groups['name'].unique()
# Filter the 'text_files' dictionary based on the values in 'names'
text_german = {key: value for key, value in text_german.items() if key[:-7] in names_german}



""" READ SANTANDER DATA """

# pict 1: countryside, pict2: couple, pict3: bridge

df_sant = pd.read_excel(path_santander+'new_merged_sentences.xlsx')
# COLUMN NAMES
column_names = {
    'Picture': 'picture',
    'Group': 'group',
    'PAR': 'name'
}
df_sant = df_sant.rename(columns=column_names)
# VALUES
column_values = {
    'picture': {'picture_1': '1', 'picture_2': '2', 'picture_3': '3'},
    'group': {'controles': 'HC', 'pacientes': 'SZ'},
}
df_sant.replace(column_values, inplace=True)
# UNIQUE NAMES
names_sant = df_sant[['name', 'group']].drop_duplicates()
names_sant.reset_index(drop=True, inplace=True)

# TEXTS TO SANTANDER TEXTS DICTIONARY
text_sant = {}
for (name, picture), group_df in df_sant.groupby(['name', 'picture']):
    print(name, picture)
    key = f'{name}_{picture}'
    utterances = group_df['utterance'].tolist()
    text_sant[key] = utterances
    


""" DELETE INTERMEDIATE VARIABLES """
del column_names, column_values, df_sant, group_df, key, name, picture, utterances



""" CENTROIDS CALCULATIONS """

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

centroids = []
results   = []
for key, lst in text_sant.items():
    print(key)
    n_sent  = len(lst)
    if n_sent <=1:
        sent_semsim = [0]  # all semsim points
        centroid    = np.zeros((768,))   # centroid
        disp        = 0  # dispersion
        sum_dist    = 0  # euclidean sum of distances
    else:
        sent_semsim = sent_sem_sim(lst)   # S-BERT / list of semsim values
        centroid    = sent_embedding_centroid(lst)[0]
        disp        = sent_embedding_centroid(lst)[1]
        sum_dist    = sent_embedding_centroid(lst)[2]
    mean_semsim = np.mean(sent_semsim).astype(float)
    max_semsim  = np.max(sent_semsim).astype(float)
    min_semsim  = np.min(sent_semsim).astype(float)
    # ssc
    ssc   = slope_sign_changes(sent_semsim)
    # number of average crossing
    n_crossing    = crossings_average(sent_semsim)[0]
    av_crossing   = crossings_average(sent_semsim)[1]
    results.append({'name': key, 'n_sent': n_sent,
                    'mean_semsim': mean_semsim,
                    'max_semsim': max_semsim,
                    'min_semsim': min_semsim,
                    'dispersion': disp,
                    'sum_dist': sum_dist})
    centroids.append({'name': key, 'centroid': centroid})


# RESULTS DATAFRAME
df = pd.DataFrame(results)
# Extract 'name' and '_bild'
df['picture'] = df['name'].str.extract(r'DS_\d+_(\d+)')
df['name'] = df['name'].str.extract(r'(.+)_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df = df.merge(names_sant, on='name', how='left')
# change positions
column_data = df.pop('group')
df.insert(1, 'group', column_data)
column_data = df.pop('picture')
df.insert(2, 'picture', column_data)
df.to_pickle(path_santander+'df_semsim_sent_santander.pkl')
# df = pd.read_pickle(path_santander+'df_semsim_sent_santander.pkl')

# CENTROIDS DATAFRAME
df_cent = pd.DataFrame(centroids)
# Extract 'name' and '_bild'
df_cent['picture'] = df_cent['name'].str.extract(r'DS_\d+_(\d+)')
df_cent['name'] = df_cent['name'].str.extract(r'(.+)_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df_cent = df_cent.merge(names_sant, on='name', how='left')
# change positions
column_data = df_cent.pop('group')
df_cent.insert(1, 'group', column_data)
column_data = df_cent.pop('picture')
df_cent.insert(2, 'picture', column_data)
df_cent.to_pickle(path_santander+'df_centroids_sent_santander.pkl')
# df_cent = pd.read_pickle(path_santander+'df_centroids_sent_santander.pkl')
   

""" FILTER RESULTS """

df_f      = df[df['n_sent'] != 1]
df_cent_f = df_cent[~((df_cent['centroid'].apply(lambda x: isinstance(x, np.ndarray) and np.all(x == 0.0))) & (df_cent['centroid'].apply(lambda x: len(x) == 768)))]


""" GRAPHS """

# SCATTER PLOTS
def scatter_plot_group(df, var1, var2, group=None, pict=None):
    # Filter the DataFrame based on 'group' and 'pict'
    if group is not None:
        df = df[df['group'] == group]
    if pict is not None:
        df = df[df['bild'] == pict]
    # Scatter plot with different colors for different groups
    sns.set_palette("colorblind")
    if group is None:
        grouped = df.groupby('group')
        for name, group_data in grouped:
            plt.scatter(group_data[var1], group_data[var2], label=f'{name}')
        plt.legend()
    else:
        plt.scatter(df[var1], df[var2])
    # Add labels and title
    plt.xlabel(f'{var1}')
    plt.ylabel(f'{var2}')
    plt.title(f'{var2} vs {var1}')
    #plt.savefig(path_results+'scatter_corr.jpg', dpi=500)
    plt.show()


# BOXPLOTS
def boxplot_group_diff(df, var1, group=None, pict=None):
    if pict is not None:
        df = df[df['bild'] == pict]
    # Exclude rows where var1 is 0
    df = df[df[var1] != 0]
    sns.set_palette("colorblind")
    sns.boxplot(x='group', y=var1, data=df)
    # Add labels and title
    plt.xlabel('Group')
    plt.ylabel(f'{var1}')
    plt.title(f'Boxplot of {var1} by Group and Picture')
    # Show the plot
    plt.show()
    
    
boxplot_group_diff(df_f, 'mean_semsim')
boxplot_group_diff(df_f, 'max_semsim')
boxplot_group_diff(df_f, 'min_semsim')
boxplot_group_diff(df_f, 'dispersion')
boxplot_group_diff(df_f, 'sum_dist')



""" UMAP - REDUCTION OF DIMENSIONALITY OF CENTROIDS SENTENCES """

centroid_array = df_cent_f['centroid'].to_numpy()
centroid_array = np.vstack(centroid_array)

tsne = TSNE(n_components=2, perplexity=10, n_jobs=-1, random_state=42)
tsne_embeddings = tsne.fit(centroid_array)
df_cent_f['tsne_embedding_1'] = tsne_embeddings[:, 0]
df_cent_f['tsne_embedding_2'] = tsne_embeddings[:, 1]

# BY PICTURE
df_cent_f_1 = df_cent_f[df_cent_f['picture'] == '1']
df_cent_f_2 = df_cent_f[df_cent_f['picture'] == '2']
df_cent_f_3 = df_cent_f[df_cent_f['picture'] == '3']


def scatter_tsne(df, division, source='all'):
    sns.set_palette("colorblind")
    sns.scatterplot(x=df['tsne_embedding_1'], y=df['tsne_embedding_2'], hue=df[division])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Dimensionality reduction - {division}')
    plt.legend(title=f'{division}', loc='upper right')
    plt.savefig(path_results_german+f'dim_reduction_{division}_{source}.jpg', dpi=500)
    # plt.show()

scatter_tsne(df_cent_f_1, 'group')
scatter_tsne(df_cent_f_2, 'group')
scatter_tsne(df_cent_f_3, 'group')

scatter_tsne(df_cent_f, 'group', 'santander')
scatter_tsne(df_cent_f, 'picture', 'santander')





""" BOTH GERMAN AND SANTANDER DATA """

# pict 1: countryside, pict2: couple, pict3: bridge

df_f      = df[df['n_sent'] != 1]
df_cent_f = df_cent[~((df_cent['centroid'].apply(lambda x: isinstance(x, np.ndarray) and np.all(x == 0.0))) & (df_cent['centroid'].apply(lambda x: len(x) == 768)))]

# ONLY PICTURE 1 and 2
df_f      = df_f[df_f['picture'] != '3']
df_cent_f = df_cent_f[df_cent_f['picture'] != '3']

# READ GERMAN DATA
df_cent_german = pd.read_pickle(path_results_german+'df_centroids_sents.pkl')

# COLUMN NAMES
column_names = {'bild': 'picture'}
df_cent_german = df_cent_german.rename(columns=column_names)
column_values = {'picture': {'1': '0', '2': '1', '4': '2'},
                 'group': {'Sz': 'SZ'}}
df_cent_german.replace(column_values, inplace=True)

# MERGE DATASETS
df_both = pd.concat([df_cent_f, df_cent_german], keys=['Spanish', 'German'])
df_both = df_both.reset_index(level=0).rename(columns={'level_0': 'source'})

# CALCUALTE T-SNE
centroid_array = df_both['centroid'].to_numpy()
centroid_array = np.vstack(centroid_array)

tsne = TSNE(n_components=2, perplexity=10, n_jobs=-1, random_state=42)
tsne_embeddings = tsne.fit(centroid_array)
df_both['tsne_embedding_1'] = tsne_embeddings[:, 0]
df_both['tsne_embedding_2'] = tsne_embeddings[:, 1]


# CLUSTERS

df_both = df_both[(df_both['picture'] == '1') | (df_both['picture'] == '2')]
df_both = df_both[df_both['group'] != 'MD']

scatter_tsne(df_both, 'source')
scatter_tsne(df_both, 'group')
scatter_tsne(df_both, 'picture')

