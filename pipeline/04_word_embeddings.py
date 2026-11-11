from pipeline.text_measures import read_files_sent, calculate_diffs, crossings_average, slope_sign_changes

from pipeline.paths import TEXTS, METADATA, RESULTS

# -*- coding: utf-8 -*-
"""
@author: Claudio
"""


# IMPORT PACKAGES
import re
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import fasttext.util
from scipy.stats import linregress

# PATH INPUT
path_input = TEXTS
path_demo = METADATA
path_results = RESULTS


def join_sentences(sentences):
    joined_text = ' '.join(sentences)
    punctuation_marks = r"[.,:;!?]"
    joined_text = re.sub(punctuation_marks, '', joined_text)
    # Split the joined text into a list of words
    word_list = re.findall(r'\b\w+\b', joined_text)
    return word_list


# FILTER BY CONTENT WORDS
def filter_content(word_list):
    # Join the list of words into a string
    text = ' '.join(word_list)
    # Process the text with spaCy
    doc = nlp(text)
    # Extract content words (tokens that are not stop words or punctuation)
    content_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return content_words


# SEMANTIC SIMILARITY CONSECUTIVE WORDS FASTTEXT
def sem_sim_fasttext(text):
    similarities = []
    # Iterate through the words in the list
    for i in range(1, len(text)):
        # Get word embeddings for consecutive words
        word1_embedding = ft.get_word_vector(text[i-1])
        word2_embedding = ft.get_word_vector(text[i])
        # Calculate cosine similarity between word embeddings
        similarity = np.dot(word1_embedding, word2_embedding) / (np.linalg.norm(word1_embedding) * np.linalg.norm(word2_embedding))
        # Append the similarity score to the list
        if not np.isnan(similarity):
            similarities.append(similarity)
    return similarities


# CENTROID AND DISPERSION OF WORDS FASTTEXT
def centroid_fasttext(text):
    embeddings = []
    for i in range(1, len(text)):
        embeddings.append(ft.get_word_vector(text[i]))
    # Calculate centroid
    centroid = np.mean(embeddings, axis=0)
    # Calculate dispersion
    dispersion = np.mean([cosine_similarity(centroid.reshape(1, -1), emb.reshape(1, -1))[0, 0] for emb in embeddings])
    # Calculate sum of euclidean distances between consecutive embeddings
    euclidean_distances = [np.linalg.norm(emb1 - emb2) for emb1, emb2 in zip(embeddings[:-1], embeddings[1:])]
    sum_distances = np.sum(euclidean_distances)
    return [centroid, dispersion, sum_distances]

def sem_sim_total(text):
    max_tokens = 500
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")['input_ids']
    num_tokens = len(tokens[0])
    if num_tokens <= max_tokens:
        return process_chunk(tokens[0])
    results = []
    for i in range(0, num_tokens, max_tokens):
        chunk = tokens[0][i:i+max_tokens]
        results.extend(process_chunk(chunk))
    return results


# PROCESS SEMANTIC SIMILARITY IN CHUNKS
def process_chunk(tokens):
    # Get word embeddings from BERT
    with torch.no_grad():
        outputs = model(tokens.unsqueeze(0))
        embeddings = outputs.hidden_states[-1]
        embeddings = embeddings[0]
    # Calculate cosine similarity between consecutive word embeddings
    # Initialize a list to store the cosine similarities
    similarities = []
    # Iterate through the embeddings
    for i in range(1, embeddings.shape[0]):
        # Get consecutive word embeddings
        word1 = embeddings[i-1]
        word2 = embeddings[i]        
        # Reshape to make sure they are of shape (embedding_dim,)
        word1 = word1.view(-1)
        word2 = word2.view(-1)
        # Calculate cosine similarity
        similarity = F.cosine_similarity(word1, word2, dim=0)
        # Append to the list of similarities
        similarities.append(similarity.item())
    return similarities


def scatter_slope(y_values, name):
    x_values = np.arange(len(y_values))
    slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)
    line_equation = np.poly1d([slope, intercept])
    line_x_values = np.linspace(0, len(y_values) - 1, 100)
    plt.scatter(x_values, y_values, label='semsim', s=15)
    plt.plot(line_x_values, line_equation(line_x_values), label='Regression Line', color='red')
    plt.xlabel('Consecutive words')
    plt.ylabel('semsim')
    plt.legend()
    plt.savefig(path_results + f'scatter_semsim_{name}.jpg', dpi=500)
    plt.close('all')
    return [slope, p_value]


# CALCULATE DIFFERENCES OF SEMSIM


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

# LOAD NLP SPACY
nlp = spacy.load("de_core_news_lg")

# LOAD FASTTEXT
fasttext.util.download_model('de', if_exists='ignore')  # German
ft = fasttext.load_model('cc.de.300.bin')

# LOAD PRE-TRAINED GERMAN BERT MODEL AND TOKENIZER
model_name = "dbmdz/bert-base-german-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

centroids = []
results   = []
semsim_hc = {}
semsim_md = {}
semsim_sz = {}
for key, lst in text_files.items():
    joined_text  = ' '.join(lst)
    sem_list_t   = sem_sim_total(joined_text)
    sem_list_t   = [x for x in sem_list_t if x < 0.9999]
    joined_list  = join_sentences(lst)
    c_words      = filter_content(joined_list)
    # euclidean sum of distances for fasttext
    word_vectors = [ft.get_word_vector(word) for word in c_words]
    points = np.array(word_vectors)
    sum_distances = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    sum_distances = sum_distances.astype(float)
    # centroid
    centroid   = centroid_fasttext(c_words)[0]
    # dispersion from centroid
    dispersion = centroid_fasttext(c_words)[1].astype(float)
    # sum distances 2
    sum_distances_2 = centroid_fasttext(c_words)[2].astype(float)
    # semsim fasttext
    sem_list     = sem_sim_fasttext(c_words)
    sem_list     = [x for x in sem_list if x < 0.9999]
    n_content    = len(sem_list)+1
    # scatterplot
    [slope, pvalue_s]  = scatter_slope(sem_list, key)
    # ssc
    ssc   = slope_sign_changes(sem_list)
    ssc_t = slope_sign_changes(sem_list_t)
    # skewness
    skew   = stats.skew(sem_list)
    skew_t = stats.skew(sem_list_t)
    # autocorrelation
    autocorr   = pd.Series(sem_list).autocorr()
    autocorr_t = pd.Series(sem_list_t).autocorr()
    # excess kurtosis
    kurt   = stats.kurtosis(sem_list)-3
    kurt_t = stats.kurtosis(sem_list_t)-3
    # mean semsim & distribution stats
    mean_semsim    = np.mean(sem_list).astype(float)
    max_semsim     = np.max(sem_list).astype(float)
    min_semsim     = np.min(sem_list).astype(float)
    p90_semsim     = np.percentile(sem_list, 90)
    p25_semsim     = np.percentile(sem_list, 25)
    p10_semsim     = np.percentile(sem_list, 10)
    mean_semsim_t  = np.mean(sem_list_t)
    max_semsim_t   = np.max(sem_list_t)
    min_semsim_t   = np.min(sem_list_t)
    p90_semsim_t   = np.percentile(sem_list_t, 90)
    p25_semsim_t   = np.percentile(sem_list_t, 25)
    p10_semsim_t   = np.percentile(sem_list_t, 10)
    # calculation for differences
    [diff, m_diff, s_diff, a_val, sum_diff, sum_a_val, m_abs_val] = calculate_diffs(sem_list)
    [diff_t, m_diff_t, s_diff_t, a_val_t, sum_diff_t, sum_a_val_t, m_abs_val_t] = calculate_diffs(sem_list_t)
    m_diff = m_diff.astype(float)
    s_diff = s_diff.astype(float)
    # distribution of differences
    min_diff   = np.min(diff).astype(float)
    max_diff   = np.max(diff).astype(float)
    p90_diff   = np.percentile(diff, 90)
    p25_diff   = np.percentile(diff, 25)
    p10_diff   = np.percentile(diff, 10)
    min_diff_t = np.min(diff_t)
    max_diff_t = np.max(diff_t)
    p90_diff_t = np.percentile(diff_t, 90)
    p25_diff_t = np.percentile(diff_t, 25)
    p10_diff_t = np.percentile(diff_t, 10)
    # number of average crossing
    n_crossing    = crossings_average(sem_list)[0]
    av_crossing   = crossings_average(sem_list)[1]
    n_crossing_t  = crossings_average(sem_list_t)[0]
    av_crossing_t = crossings_average(sem_list_t)[1]
    # Extract group information from the 'groups' DataFrame
    group_info = groups[groups['name'] == key[:-7]]['group'].values[0]
    # Determine the target dictionary based on the group
    target_dict = None
    if group_info == 'HC':
        target_dict = semsim_hc
    elif group_info == 'MD':
        target_dict = semsim_md
    elif group_info == 'SZ':
        target_dict = semsim_sz
    # If a valid group is found, update the target dictionary
    if target_dict is not None:
        key_for_dict = key[-1]
        target_dict[key_for_dict] = sem_list
    # parameters of semsim distribution
    #mu           = fit_normal_distribution(sem_list)[0]
    #std          = fit_normal_distribution(sem_list)[1]
    #shape        = fit_lognormal_distribution(sem_list)[0]
    #loc          = fit_lognormal_distribution(sem_list)[1]
    #scale        = fit_lognormal_distribution(sem_list)[2]
    # fit and goodnes-of-fit
    # [ks_n, ks_p_n, ks_logn, ks_p_logn] = fit_and_plot_distributions(sem_list, key)
    results.append({'name': key, 'n_content': n_content, 'slope': slope, 'pvalue_s': pvalue_s,
                    'sum_distances': sum_distances, 'sum_distances_n': sum_distances/n_content,
                    'sum_distances_2': sum_distances_2, 'dispersion': dispersion,
                    'mean_semsim': mean_semsim, 'min_semsim': min_semsim, 'max_semsim': max_semsim,
                    'p90_semsim': p90_semsim, 'p25_semsim': p25_semsim, 'p10_semsim': p10_semsim,
                    'av_ssc': ssc/n_content, 'skew': skew, 'autocorr' : autocorr, 'kurt': kurt,
                    'n_crossing': n_crossing, 'av_crossing': av_crossing,
                    'mean_diff': m_diff, 'std_diff': s_diff,
                    'sum_diff': sum_diff, 'mean_abs_diff': m_abs_val,
                    'min_diff': min_diff, 'max_diff': max_diff,
                    'p90_diff': p90_diff, 'p25_diff': p25_diff, 'p10_diff': p10_diff,
                    'mean_semsim_t': mean_semsim_t, 'min_semsim_t': min_semsim_t, 'max_semsim_t': max_semsim_t,
                    'p90_semsim_t': p90_semsim_t, 'p25_semsim_t': p25_semsim_t, 'p10_semsim_t': p10_semsim_t,
                    'av_ssc_t': ssc_t/n_content, 'skew_t': skew_t, 'autocorr_t' : autocorr_t, 'kurt_t': kurt_t,
                    'n_crossing_t': n_crossing_t, 'av_crossing_t': av_crossing_t,
                    'mean_diff_t': m_diff_t, 'std_diff_t': s_diff_t,
                    'sum_diff_t': sum_diff_t, 'mean_abs_diff_t': m_abs_val_t,
                    'min_diff_t': min_diff_t, 'max_diff_t': max_diff_t,
                    'p90_diff_t': p90_diff_t, 'p25_diff_t': p25_diff_t, 'p10_diff_t': p10_diff_t
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
# save to pickle file
df.to_pickle(path_results+'df_semsim_dist_length.pkl')
# df = pd.read_pickle(path_results+'df_semsim_dist_length.pkl')


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
df_cent.to_pickle(path_results+'df_centroids_words.pkl')
# df_cent = pd.read_pickle(path_results+'df_centroids_words.pkl')

