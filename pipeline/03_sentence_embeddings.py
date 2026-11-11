from pipeline.text_measures import read_files_sent, calculate_diffs, crossings_average, slope_sign_changes

from pipeline.paths import TEXTS, METADATA, RESULTS

# -*- coding: utf-8 -*-
"""
@author: Claudio
"""


# IMPORT PACKAGES
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# PATH INPUT
path_input = TEXTS
path_demo = METADATA
path_results = RESULTS

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def sent_sem_sim(sentences):
    # model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') # 768 
    encoded_input = sbert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    #sentence_embeddings = model.encode(sentences)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    all_similarities = []
    for i in range(len(sentences)-1):
        emb1 = sentence_embeddings[i].reshape(1, -1)
        emb2 = sentence_embeddings[i+1].reshape(1, -1)
        similarity_matrix = cosine_similarity(emb1, emb2)
        all_similarities.append(similarity_matrix[0, 0])
    return all_similarities


def sent_embedding_centroid(sentences):
    # model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') # 768 dimensions
    # sentence_embeddings = model.encode(sentences)
    # Tokenize sentences
    encoded_input = sbert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    #sentence_embeddings = model.encode(sentences)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).detach().cpu().numpy()
    # Calculate centroid
    centroid = np.mean(sentence_embeddings, axis=0)
    # Calculate dispersion
    dispersion = np.mean([cosine_similarity(centroid.reshape(1, -1), emb.reshape(1, -1))[0, 0] for emb in sentence_embeddings])
    # Calculate sum of euclidean distances between consecutive embeddings
    euclidean_distances = [np.linalg.norm(emb1 - emb2) for emb1, emb2 in zip(sentence_embeddings[:-1], sentence_embeddings[1:])]
    sum_distances = np.sum(euclidean_distances)
    return [centroid, dispersion, sum_distances]


# CALCULATE NUMBER OF TIMES CROSSING THE AVERAGE


# SLOPE SIGN CHANGES


text_files = read_files_sent(path_input)
text_files = {key[:-4]: value for key, value in text_files.items()}
# IMPORT DEMOGRAPHICS
groups = pd.read_excel(path_demo+"groups.xlsx")
# UNIQUE NAMES
names = groups['name'].unique()
# Filter the 'text_files' dictionary based on the values in 'names'
text_files = {key: value for key, value in text_files.items() if key[:-7] in names}

stats_length = pd.read_pickle(path_results+'df_stats_length.pkl')

#model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
sbert_tokenizer = AutoTokenizer.from_pretrained('aari1995/German_Semantic_STS_V2')
model           = AutoModel.from_pretrained('aari1995/German_Semantic_STS_V2')


centroids = []
results   = []
for key, lst in text_files.items():
    print(key)
    n_sent  = len(lst)
    if n_sent <=1:
        sent_semsim = 0  # all semsim points
        disp        = 0  # dispersion
        sum_dist    = 0  # euclidean sum of distances
        centroid    = np.zeros((1024,))   # centroid
    else:
        sent_semsim = sent_sem_sim(lst)   # S-BERT / list of semsim values
        disp        = sent_embedding_centroid(lst)[1]
        sum_dist    = sent_embedding_centroid(lst)[2]
        centroid    = sent_embedding_centroid(lst)[0]
    mean_semsim = np.mean(sent_semsim).astype(float)
    max_semsim  = np.max(sent_semsim).astype(float)
    min_semsim  = np.min(sent_semsim).astype(float)
    p90_semsim  = np.percentile(sent_semsim, 90)
    p25_semsim  = np.percentile(sent_semsim, 25)
    p10_semsim  = np.percentile(sent_semsim, 10)
    # ssc
    if n_sent <=1:
        ssc = np.nan
    else:
        ssc   = slope_sign_changes(sent_semsim)
    # autocorrelation semsim
    if n_sent>2:
        autocorr = pd.Series(sent_semsim).autocorr()
    else:
        autocorr = np.nan
    # number of average crossing
    if n_sent>1:
        n_crossing    = crossings_average(sent_semsim)[0]
        av_crossing   = crossings_average(sent_semsim)[1]
    else:
        n_crossing    = np.nan
        av_crossing   = np.nan
    # calculation for differences
    if n_sent>1:
        [diff, m_diff, s_diff, a_val, sum_diff, sum_a_val, m_abs_val] = calculate_diffs(sent_semsim)
        m_diff = m_diff.astype(float)
        s_diff = s_diff.astype(float)
    else:
        m_diff = np.nan
        s_diff = np.nan
    results.append({'name': key, 'n_sent': n_sent,
                    'dispersion': disp, 'sum_dist': sum_dist, 'sum_dist_n': sum_dist/n_sent,
                    'max_semsim': max_semsim,
                    'mean_semsim': mean_semsim, 'min_semsim': min_semsim,
                    'p90_semsim': p90_semsim, 'p25_semsim': p25_semsim, 'p10_semsim': p10_semsim,
                    'ssc': ssc, 'autocorr': autocorr, 'n_crossing': n_crossing, 'av_crossing': av_crossing,
                    'mean_diff': m_diff, 'std_diff': s_diff, 'sum_diff': sum_diff,
                    'mean_abs_diff': m_abs_val
                    })
    centroids.append({'name': key, 'centroid': centroid})
    

# RESULTS DATAFRAME
df = pd.DataFrame(results)
# Extract 'name' and '_bild'
df['bild'] = df['name'].str.extract(r'_bild_(\d+)')
df['name'] = df['name'].str.extract(r'(.+)_bild_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df = df.merge(groups, on='name', how='left')
# Remove columns
df = df.drop(columns=['s_group', 'type'])
# change positions
column_data = df.pop('group')
df.insert(1, 'group', column_data)
column_data = df.pop('bild')
df.insert(2, 'bild', column_data)
# merge
df = pd.merge(df, stats_length[['name', 'group', 'bild', 'total_length', 'sent_av_length']], on=['name', 'group', 'bild'], how='left')
column_data = df.pop('total_length')
df.insert(3, 'total_length', column_data)
column_data = df.pop('sent_av_length')
df.insert(4, 'sent_av_length', column_data)
df.to_pickle(path_results+'df_semsim_sent_length.pkl')
# df = pd.read_pickle(path_results+'df_semsim_sent_length.pkl')


# CENTROIDS DATAFRAME
df_cent = pd.DataFrame(centroids)
# Extract 'name' and '_bild'
df_cent['bild'] = df_cent['name'].str.extract(r'_bild_(\d+)')
df_cent['name'] = df_cent['name'].str.extract(r'(.+)_bild_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df_cent = df_cent.merge(groups, on='name', how='left')
# Remove columns
df_cent = df_cent.drop(columns=['s_group', 'type'])
# change positions
column_data = df_cent.pop('group')
df_cent.insert(1, 'group', column_data)
column_data = df_cent.pop('bild')
df_cent.insert(2, 'bild', column_data)
df_cent.to_pickle(path_results+'df_centroids_sents_mono.pkl')

