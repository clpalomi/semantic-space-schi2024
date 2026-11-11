from pipeline.text_measures import read_files_sent, sentence_lengths, length_statistics, crossings_average

# Additional analysis; check additional/README.md before running.
from pipeline.paths import TEXTS, METADATA, ADDITIONAL_RESULTS

# -*- coding: utf-8 -*-
"""
@author: Claudio
"""
# INSTALL

# pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.6.0/de_core_news_lg-3.6.0-py3-none-any.whl
# pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.6.0/de_core_news_lg-3.6.0.tar.gz (THIS ONE)


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
from itertools import combinations
from scipy.stats import ttest_ind



# PATH INPUT
path_input = TEXTS
path_demo = METADATA
path_results = ADDITIONAL_RESULTS

# EXCLUDE 0401_3


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


# SEMANTIC SIMILARITY
def sem_sim(text):
    text = ' '.join(text)
    max_tokens = 500
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")['input_ids']
    num_tokens = len(tokens[0])
    if num_tokens <= max_tokens:
        return process_chunk(tokens)
    results = []
    for i in range(0, num_tokens, max_tokens):
        chunk = tokens[i:i+max_tokens]
        results.extend(process_chunk(chunk))
    return results


# PROCESS SEMANTIC SIMILARITY IN CHUNKS
def process_chunk(tokens):
    # Get word embeddings from BERT
    with torch.no_grad():
        outputs = model(tokens)
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
    slope, intercept = np.polyfit(x_values, y_values, 1)
    # Create a line equation using the slope and intercept
    line_equation = np.poly1d([slope, intercept])
    # Generate x values for the line
    line_x_values = np.linspace(0, len(y_values) - 1, 100)
    # Plot scatter plot of the values
    plt.scatter(x_values, y_values, label='semsim', s=15)
    # Plot the line
    plt.plot(line_x_values, line_equation(line_x_values), label='Regression Line', color='red')
    plt.xlabel('Consecutive words')
    plt.ylabel('semsim')
    plt.legend()
    plt.savefig(path_results+f'scatter_semsim_{name}.jpg', dpi=500)
    # Close all open figures
    plt.close('all')
    return slope


# CALCULATE DIFFERENCES OF SEMSIM
def calculate_diffs(sem_list):
    # Calculate first differences
    differences = [sem_list[i] - sem_list[i - 1] for i in range(1, len(sem_list))]
    # mean and std
    m_diff = np.mean(differences)
    s_diff = np.std(differences)
    # Calculate absolute values
    absolute_values = [abs(diff) for diff in differences]
    # Calculate sum of differences
    sum_differences = sum(differences)
    # Calculate sum of absolute values
    sum_absolute_values = sum(absolute_values)
    # Calculate the mean of absolute values
    mean_absolute_values = sum_absolute_values / len(absolute_values) if absolute_values else 0
    return [differences, m_diff, s_diff,
            absolute_values, sum_differences, 
            sum_absolute_values, mean_absolute_values]


# CALCULATE NUMBER OF TIMES CROSSING THE AVERAGE



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


""" SEMANTICS: LANGUAGE MODEL TO LOAD: BERT """

# LOAD PRE-TRAINED GERMAN GPT-2 MODEL AND TOKENIZER
model_name = "dbmdz/bert-base-german-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

# LOAD NLP SPACY
nlp = spacy.load("de_core_news_lg")


""" SEMANTIC SIMILARITTY WITHIN SENTENCES """


""" SEMANTIC SIMILARITTY BETWEEN SENTENCES """


""" SEMANTIC SIMILARITTY CONSECUTIVE WORDS """

# 1.1 distribution (all the points) (by person)

results = []
semsim_hc = {}
semsim_md = {}
semsim_sz = {}
for key, lst in text_files.items():
    joined_list  = join_sentences(lst)
    c_words      = filter_content(joined_list)
    n_content    = len(c_words)
    sem_list     = sem_sim(c_words)
    # scatterplot
    slope        = scatter_slope(sem_list, key)
    # mean semsim
    mean_semsim  = np.mean(sem_list)
    # calculation for differences
    [diff, m_diff, s_diff, a_val, sum_diff, sum_a_val, m_abs_val] = calculate_diffs(sem_list)
    # number of average crossing
    n_crossing  = crossings_average(sem_list)[0]
    av_crossing = crossings_average(sem_list)[1]
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
    mu           = fit_normal_distribution(sem_list)[0]
    std          = fit_normal_distribution(sem_list)[1]
    shape        = fit_lognormal_distribution(sem_list)[0]
    loc          = fit_lognormal_distribution(sem_list)[1]
    scale        = fit_lognormal_distribution(sem_list)[2]
    # fit and goodnes-of-fit
    [ks_n, ks_p_n, ks_logn, ks_p_logn] = fit_and_plot_distributions(sem_list, key)
    results.append({'name': key, 'n content': n_content, 'mean semsim': mean_semsim,
                    'mean diff': m_diff, 'std diff': s_diff,
                    'sum diff': sum_diff, 'mean abs diff': m_abs_val,
                    'n crossing': n_crossing, 'av crossing': av_crossing,
                    'mu': mu, 'std': std,
                    'shape': shape, 'loc': loc, 'scale': scale,
                    'ks_n': ks_n, 'ks_p_n': ks_p_n,
                    'ks_logn': ks_logn, 'ks_p_logn': ks_p_logn})


# DataFrame from results list
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
    plt.scatter(df[var1], df[var2])
    # Add labels and title
    plt.xlabel(f'{var1}')
    plt.ylabel(f'{var2}')
    plt.title(f'{var2} vs {var1}')
    plt.show()


# SCATTER PLOTS

scatter_plot(df, 'total_length', 'n content')
scatter_plot(df, 'total_length', 'sent_av_length')
scatter_plot(df, 'total_length', 'mean semsim')       # !

scatter_plot(df, 'n content', 'mean semsim')          # !
scatter_plot(df, 'n content', 'mean diff')            # !!
scatter_plot(df, 'n content', 'std diff')

scatter_plot(df, 'mean diff', 'std diff')

scatter_plot(df, 'n content', 'sum diff')
scatter_plot(df, 'n content', 'mean abs diff')
scatter_plot(df, 'n content', 'n crossing')
scatter_plot(df, 'n content', 'av crossing')
scatter_plot(df, 'n content', 'ks_n')
scatter_plot(df, 'n content', 'ks_p_n')
scatter_plot(df, 'n content', 'ks_logn')
scatter_plot(df, 'n content', 'ks_p_logn')

scatter_plot(df, 'mean semsim', 'mean diff')
scatter_plot(df, 'mean semsim', 'sum diff')
scatter_plot(df, 'mean semsim', 'mean abs diff')
scatter_plot(df, 'mean semsim', 'n crossing')
scatter_plot(df, 'mean semsim', 'av crossing')


# SCATTER PLOT BY GROUP AND PICTURE

scatter_plot_group(df, 'n content', 'mean semsim', group='HC', pict='1')
scatter_plot_group(df, 'n content', 'mean semsim', group='HC', pict='2')
scatter_plot_group(df, 'n content', 'mean semsim', group='HC', pict='4')
scatter_plot_group(df, 'n content', 'mean semsim', group='HC', pict='6')






# 1.2 distribution by group and picture
# 1.3 distribution by group


