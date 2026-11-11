from pipeline.text_measures import read_files_sent, sentence_lengths, length_statistics, calculate_diffs, crossings_average, slope_sign_changes

# Additional analysis; check additional/README.md before running.
from pipeline.paths import TEXTS, METADATA, ADDITIONAL_RESULTS

# -*- coding: utf-8 -*-
"""
@author: Claudio
"""
# INSTALL

# pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.6.0/de_core_news_lg-3.6.0-py3-none-any.whl
# pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.6.0/de_core_news_lg-3.6.0.tar.gz (THIS ONE)
# pip install fasttext-wheel
# pip install sentence-transformers
# pip install umap-learn


# IMPORT PACKAGES
import os
import re
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import time
import seaborn as sns
import spacy
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import itertools
from scipy.stats import norm, lognorm, probplot, kstest
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from scipy.stats import ttest_ind
import fasttext.util
from scipy.stats import linregress
from sentence_transformers import SentenceTransformer

# PATH INPUT
path_input = TEXTS
path_demo = METADATA
path_results = ADDITIONAL_RESULTS



""" FUNCTIONS"""

# CREATE A DICTIONARY OF TEXT FROM FOLDER


# CALCULATE SENTENCE LENGTHS


# CALCULATE LENGTH STATISTICS


# GRAPH COMPARISON
def boxplot_comparison(df, hue_col='group', numerical_cols=['total_length', 'sent_av_length', 'sent_std_length']):
    # Set seaborn style for better visualization
    sns.set(style="whitegrid")
    # Loop through unique 'bild' values
    for bild_value in df['bild'].unique():
        # Create a subset of the DataFrame for the current 'bild' value
        subset_df = df[df['bild'] == bild_value]
        # Initialize the matplotlib figure
        plt.figure(figsize=(15, 8))
        # Loop through numerical columns and create subplots for each
        for i, col in enumerate(numerical_cols, 1):
            plt.subplot(1, len(numerical_cols), i)
            # Create a boxplot for the current 'bild' with different colors for each 'group'
            ax = sns.boxplot(x=hue_col, y=col, data=subset_df)
            # Set plot title
            plt.title(f'{col} - Picture {bild_value} - by {hue_col}', fontsize=14)
            # Set x-axis and y-axis label size
            ax.set_xlabel(hue_col.capitalize(), fontsize=13)
            ax.set_ylabel(col.capitalize(), fontsize=13)
            # Perform statistical tests and annotate significant differences
            group_pairs = list(itertools.combinations(subset_df[hue_col].unique(), 2))
            p_values = []
            for pair in group_pairs:
                group1 = subset_df[subset_df[hue_col] == pair[0]][col]
                group2 = subset_df[subset_df[hue_col] == pair[1]][col]
                # Perform statistical test (e.g., t-test)
                _, p_value = stats.ttest_ind(group1, group2)
                p_values.append(p_value)
            # Perform multiple testing correction
            corrected_p_values = multipletests(p_values, method='bonferroni')[1]
            # Use Annotator to show significant differences
            annotator = Annotator(ax, group_pairs, data=subset_df, x=hue_col, y=col)
            annotator.set_pvalues(corrected_p_values)
            annotator.annotate()
        plt.tight_layout()
        plt.savefig(path_results+f'length_stats_picture{bild_value}.jpg', dpi=1000)
        plt.show()


# JOIN SENTENCES OF TEXT_FILES
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


# SEMANTIC SIMILARITY BETWEEN SENTENCES (SENTENCE TRANSFORMER BERT)
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


# SEMANTIC SIMILARITY ALL WORDS (BERT)
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


# FUNCTION TO FIT A NORMAL DISTRIBUTION
def fit_normal_distribution(data):
    mu, std = norm.fit(data)
    return mu, std


# FUNCTION TO FIT A LOG-NORMAL DISTRIBUTION
def fit_lognormal_distribution(data):
    params = lognorm.fit(data)
    return params


# FIT AND PLOT DISTRIBUTIONS
def fit_and_plot_distributions(values, name):
    # Plot histogram
    plt.hist(values, bins=20, density=True, alpha=0.6, color='g', edgecolor='black')
    # Fit a normal distribution
    mu, std = fit_normal_distribution(values)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    # Fit a log-normal distribution
    params = fit_lognormal_distribution(values)
    lognorm_dist = lognorm(*params)
    p = lognorm_dist.pdf(x)
    plt.plot(x, p, 'r', linewidth=2)
    # Add labels and title
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Fitted Distributions')
    plt.savefig(path_results+f'distribution_semsim_{name}.jpg', dpi=500)
    # Q-Q plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    probplot(values, plot=plt)
    plt.title('Q-Q Plot for Normal Distribution')
    plt.subplot(1, 2, 2)
    probplot(values, dist=lognorm, sparams=params, plot=plt)
    plt.title('Q-Q Plot for Log-Normal Distribution')
    plt.savefig(path_results+f'distribution_semsim_{name}_fit.jpg', dpi=500)
    # Goodness-of-fit tests
    ks_stat_normal, ks_p_value_normal = kstest(values, 'norm', args=(mu, std))
    ks_stat_lognorm, ks_p_value_lognorm = kstest(values, 'lognorm', args=params)
    print("\nGoodness-of-fit results for Normal Distribution:")
    print(f"KS Statistic: {ks_stat_normal}, p-value: {ks_p_value_normal}")
    print("\nGoodness-of-fit results for Log-Normal Distribution:")
    print(f"KS Statistic: {ks_stat_lognorm}, p-value: {ks_p_value_lognorm}")
    # Close all open figures
    plt.close('all')
    return [ks_stat_normal, ks_p_value_normal, ks_stat_lognorm, ks_p_value_lognorm]


# SCATTER PLOT AND SLOPE OF CONSECUTIVE POINTS
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



""" PROCESSING: SPLIT SPEECH INTO SENTENCES """

# SAVE EACH LIST OF WORDS
text_files = read_files_sent(path_input)
text_files = {key[:-4]: value for key, value in text_files.items()}
# IMPORT DEMOGRAPHICS
groups = pd.read_excel(path_demo+"groups.xlsx")
# UNIQUE NAMES
names = groups['name'].unique()
# Filter the 'text_files' dictionary based on the values in 'names'
text_files = {key: value for key, value in text_files.items() if key[:-7] in names}



""" LENGTH STATISTICS """

sent_lengths = sentence_lengths(text_files)
stats_length = length_statistics(sent_lengths)

# Extract 'name' and '_bild'
stats_length['bild'] = stats_length['filename'].str.extract(r'_bild_(\d+)')
stats_length['name'] = stats_length['filename'].str.extract(r'(.+)_bild_\d+')

# Merge the 'groups' DataFrame based on matching values in the 'name' column
stats_length = stats_length.merge(groups, on='name', how='left')

# stats to excel and pickle file
stats_length = stats_length[['name', 'bild', 'group', 'total_length',
                            'sent_av_length', 'sent_std_length']]
stats_length.to_excel(path_results+'stats_length.xlsx', index=False)



""" COMPARISON BY PICTURE """

# COMPARISONS BOXPLOTS
boxplot_comparison(stats_length, numerical_cols=['total_length', 'sent_av_length'])

# SUMMARY OF STATS LENGTH
summ_stats_length = stats_length.groupby(['bild', 'group']).agg({
    'total_length': ['mean', 'std'],
    'sent_av_length': ['mean', 'std']
}).reset_index()
summ_stats_length.columns = ['_'.join(col).strip() for col in summ_stats_length.columns.values]
summ_stats_length = summ_stats_length.rename(columns={
    'total_length_mean': 'mean_total_length',
    'total_length_std': 'std_total_length',
    'sent_av_length_mean': 'mean_sent_av_length',
    'sent_av_length_std': 'std_sent_av_length'
})
summ_stats_length.to_excel(path_results+'summ_stats_length.xlsx', index=False)

# SUMMARY DIFFERENCES
significance_data = []
# List of variables to analyze
variables = ['total_length', 'sent_av_length']
# Iterate through variables and bilds
for variable in variables:
    for bild in stats_length['bild'].unique():
        # Get data for the current bild
        bild_data = stats_length[stats_length['bild'] == bild]
        # Perform t-tests for all possible pairwise comparisons
        groups = bild_data['group'].unique()
        group_combinations = list(combinations(groups, 2))
        for group1, group2 in group_combinations:
            data_group1 = bild_data[bild_data['group'] == group1][variable]
            data_group2 = bild_data[bild_data['group'] == group2][variable]
            # Perform t-test
            _, p_value = ttest_ind(data_group1, data_group2)
            # Bonferroni correction
            p_value_corrected = p_value * len(group_combinations)
            # Append results to the significance dataframe
            # Store results in a dictionary
            result_dict = {
                'bild': bild,
                'variable': variable,
                'group1': group1,
                'group2': group2,
                'p_value': p_value,
                'p_value_corrected': p_value_corrected
            }
            significance_data.append(result_dict)

significance_df = pd.concat([pd.DataFrame([result]) for result in significance_data], ignore_index=True)
significance_df = significance_df[['variable', 'bild', 'group1', 'group2',
                                   'p_value', 'p_value_corrected']]
significance_df.to_excel(path_results+'pvalue_stats_length.xlsx', index=False)



""" DELETE INTERMEDIATE VARIABLES """

del bild, bild_data, data_group1, data_group2, group1, group2, group_combinations
del p_value, p_value_corrected, groups
del result_dict, significance_data, variable, variables
groups = pd.read_excel(path_demo+"groups.xlsx")



""" SEMANTIC SIMILARITTY BETWEEN SENTENCES """

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


centroids = []
results   = []
for key, lst in text_files.items():
    print(key)
    n_sent  = len(lst)
    if n_sent <=1:
        sent_semsim = 0  # all semsim points
        disp        = 0  # dispersion
        sum_dist    = 0  # euclidean sum of distances
        centroid    = np.zeros((768,))   # centroid
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
    ssc   = slope_sign_changes(sent_semsim)
    # autocorrelation
    autocorr = pd.Series(sent_semsim).autocorr()
    # number of average crossing
    n_crossing    = crossings_average(sent_semsim)[0]
    av_crossing   = crossings_average(sent_semsim)[1]
    # calculation for differences
    [diff, m_diff, s_diff, a_val, sum_diff, sum_a_val, m_abs_val] = calculate_diffs(sent_semsim)
    m_diff = m_diff.astype(float)
    s_diff = s_diff.astype(float)
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
df_cent.to_pickle(path_results+'df_centroids_sents.pkl')



""" SEMANTIC SIMILARITTY CONSECUTIVE WORDS """

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



""" GRAPHS AND REGRESSIONS """

def scatter_plot(df, var1, var2):
    plt.scatter(df[var1], df[var2])
    # Add labels and title
    plt.xlabel(f'{var1}')
    plt.ylabel(f'{var2}')
    plt.title(f'{var2} vs {var1}')
    plt.show()
    
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
    
    
# SENTENCES
scatter_plot_group(df, 'n_sent', 'max_semsim')
scatter_plot_group(df, 'n_sent', 'mean_semsim')
scatter_plot_group(df, 'n_sent', 'min_semsim')
scatter_plot_group(df, 'n_sent', 'dispersion')
scatter_plot_group(df, 'n_sent', 'sum_dist')
scatter_plot_group(df, 'n_sent', 'n_crossing')
scatter_plot_group(df, 'n_sent', 'av_crossing')
scatter_plot_group(df, 'n_sent', 'mean_diff')
scatter_plot_group(df, 'n_sent', 'std_diff')
scatter_plot_group(df, 'n_sent', 'sum_diff')
scatter_plot_group(df, 'n_sent', 'mean_abs_diff')



# SCATTER PLOTS
scatter_plot_group(df, 'n_content', 'max_semsim')
scatter_plot_group(df, 'n_content', 'min_semsim')
scatter_plot_group(df, 'n_content', 'mean_semsim')
scatter_plot_group(df, 'total_length', 'mean_semsim_t')
scatter_plot_group(df, 'n_content', 'av_ssc')
scatter_plot_group(df, 'n_content', 'skew')
scatter_plot_group(df, 'n_content', 'autocorr')
scatter_plot_group(df, 'n_content', 'kurt')
scatter_plot_group(df, 'n_content', 'av_crossing')
scatter_plot_group(df, 'n_content', 'mean_abs_diff')
scatter_plot_group(df, 'n_content', 'max_diff')
scatter_plot_group(df, 'n_content', 'min_diff')


scatter_plot_group(df, 'total_length', 'mean_semsim')


var1 = 'total_length'
var2_list = ['n_content', 'sent_av_length', 'mean_semsim_t', 'min_semsim_t', 'max_semsim_t',
             'mean_diff_t', 'mean_abs_diff_t']
for var2 in var2_list:
    scatter_plot_group(df, var1, var2)
    
var1 = 'n_content'
var2_list = ['mean_semsim', 'min_semsim', 'p90_semsim',
             'p10_semsim', 'n_crossing', 'av_crossing', 'mean_diff', 'std_diff',
             'sum_diff', 'mean_abs_diff', 'min_diff', 'max_diff', 'p90_diff',
             'p25_diff', 'p10_diff']
for var2 in var2_list:
    scatter_plot_group(df, var1, var2)
    
    
var1 = 'mean_semsim'
var2_list = ['av_crossing', 'mean_diff', 'sum_diff', 'mean_abs_diff',
             'min_diff', 'max_diff', 'p90_diff']
for var2 in var2_list:
    scatter_plot_group(df, var1, var2)

scatter_plot_group(df, 'mean_semsim', 'mean_abs_diff')  #


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
    
# SENTENCES
boxplot_group_diff(df, 'dispersion')
boxplot_group_diff(df, 'sum_dist')

boxplot_group_diff(df, 'mean_semsim')
boxplot_group_diff(df, 'max_semsim')
boxplot_group_diff(df, 'min_semsim')
boxplot_group_diff(df, 'p90_semsim')
boxplot_group_diff(df, 'p10_semsim')
boxplot_group_diff(df, 'ssc')
boxplot_group_diff(df, 'autocorr')
boxplot_group_diff(df, 'n_crossing')
boxplot_group_diff(df, 'av_crossing')
boxplot_group_diff(df, 'mean_diff')
boxplot_group_diff(df, 'std_diff')
boxplot_group_diff(df, 'sum_diff')
boxplot_group_diff(df, 'mean_abs_diff')



boxplot_group_diff(df, 'sum_distances_n')
boxplot_group_diff(df, 'sum_distances')

boxplot_group_diff(df, 'slope')
boxplot_group_diff(df, 'total_length')
boxplot_group_diff(df, 'mean_semsim')
boxplot_group_diff(df, 'max_semsim')
boxplot_group_diff(df, 'min_semsim')
boxplot_group_diff(df, 'p90_semsim')
boxplot_group_diff(df, 'p25_semsim')
boxplot_group_diff(df, 'p10_semsim')
boxplot_group_diff(df, 'av_ssc')
boxplot_group_diff(df, 'skew')
boxplot_group_diff(df, 'autocorr')
boxplot_group_diff(df, 'kurt')
boxplot_group_diff(df, 'av_crossing')
boxplot_group_diff(df, 'mean_semsim_t')
boxplot_group_diff(df, 'mean diff t')
boxplot_group_diff(df, 'std diff t')
boxplot_group_diff(df, 'sum diff t')
boxplot_group_diff(df, 'mean abs diff t')
boxplot_group_diff(df, 'av_crossing_t')
boxplot_group_diff(df, 'mean semsim')
boxplot_group_diff(df, 'mean diff')
boxplot_group_diff(df, 'std diff')
boxplot_group_diff(df, 'sum diff')
boxplot_group_diff(df, 'mean abs diff')
boxplot_group_diff(df, 'av crossing')



## MIXED MODEL

from statsmodels.regression.mixed_linear_model import MixedLM

def mixed_model_results(y, x):
    model_r = MixedLM.from_formula(f'{y} ~ group + {x} + sent_av_length + bild', df, groups=df["name"])
    result  = model_r.fit()
    print(result.summary())


# FASTTEXT
mixed_model_results('sum_distances', 'n_content')    # SZ (+)
mixed_model_results('sum_distances_n', 'n_content')  # SZ (+)
mixed_model_results('max_semsim', 'n_content')     # MD (-)
mixed_model_results('min_semsim', 'n_content')
mixed_model_results('mean_semsim', 'n_content')
mixed_model_results('p90_semsim', 'n_content')
mixed_model_results('p25_semsim', 'n_content')
mixed_model_results('av_crossing', 'n_content')    # MD (-)
mixed_model_results('av_ssc', 'n_content')
mixed_model_results('autocorr', 'n_content')       # MD (+)
mixed_model_results('mean_diff', 'n_content')
mixed_model_results('std_diff', 'n_content')       # MD (-)
mixed_model_results('mean_abs_diff', 'n_content')  # MD (-)
mixed_model_results('min_diff', 'n_content')
mixed_model_results('max_diff', 'n_content')       # MD (-)
mixed_model_results('p90_diff', 'n_content')       # MD (-)
mixed_model_results('p25_diff', 'n_content')       # MD (+) / SZ (+)
mixed_model_results('p10_diff', 'n_content')


# BERT
mixed_model_results('max_semsim_t', 'total_length')     # SZ (+)
mixed_model_results('mean_semsim_t', 'total_length')
mixed_model_results('p90_semsim_t', 'total_length')
mixed_model_results('p25_semsim_t', 'total_length')
mixed_model_results('av_crossing_t', 'total_length')    # SZ (-)
mixed_model_results('av_ssc_t', 'total_length')         # SZ (-)
mixed_model_results('autocorr_t', 'total_length')       # SZ (+)
mixed_model_results('mean_diff_t', 'total_length')
mixed_model_results('std_diff_t', 'total_length')
mixed_model_results('mean_abs_diff_t', 'total_length')
mixed_model_results('min_diff_t', 'total_length')
mixed_model_results('max_diff_t', 'total_length')
mixed_model_results('p90_diff_t', 'total_length')
mixed_model_results('p25_diff_t', 'total_length')
mixed_model_results('p10_diff_t', 'total_length')


# SENTENCE BERT
mixed_model_results('mean_semsim', 'n_sent')   # T2 (-) / T4 (+) / T6 (+)
mixed_model_results('max_semsim', 'n_sent')
mixed_model_results('min_semsim', 'n_sent')
mixed_model_results('ssc', 'n_sent')            # n sent (+)
mixed_model_results('av_crossing', 'n_sent')
mixed_model_results('mean_diff', 'n_sent')       # MD (+)
mixed_model_results('mean_abs_diff', 'n_sent')






    
