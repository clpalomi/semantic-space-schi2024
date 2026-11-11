from pipeline.paths import METADATA, RESULTS, TABLES

"""
@author: Claudio

Tables and results used in the article

"""

import pandas as pd

# PATH
path_demo = METADATA
path_tables = TABLES
path_results = RESULTS

""" 7. TABLE 4 - VARIABLES DIFFERENCES """

from scipy.stats import kruskal

df_words = pd.read_pickle(path_results+'df_semsim_dist_length.pkl')
#df_sents = pd.read_pickle(path_results+'df_semsim_sent_length.pkl')

# Run Kruskal-Wallis test
def kw_comparison(df, var):
    groups     = df['group'].unique()
    group_data = [
    df[df['group'] == group][var].dropna()  # Filter out NaN values
        for group in groups
        ]
    statistic, p_value = kruskal(*group_data)
    print("Kruskal-Wallis Test:")
    print("H-statistic:", statistic)
    print("p-value:", p_value)
    return [statistic, p_value]
    

# FASTTEXT
kw_comparison(df_words, 'mean_semsim')   # n.s.
kw_comparison(df_words, 'max_semsim')    # 0.0403
kw_comparison(df_words, 'min_semsim')    # n.s.

kw_comparison(df_words, 'av_ssc')          # n.s.
kw_comparison(df_words, 'av_crossing')     # p-value: 0.004081127750315587
kw_comparison(df_words, 'autocorr')        # p-value: 0.015068247668205203
kw_comparison(df_words, 'skew')            # p-value: 0.039739871

# BERT
kw_comparison(df_words, 'mean_semsim_t')   # n.s.
kw_comparison(df_words, 'max_semsim_t')    # n.s.
kw_comparison(df_words, 'min_semsim_t')    # n.s.

kw_comparison(df_words, 'av_ssc_t')          # p-value: 0.012573279532508995
kw_comparison(df_words, 'av_crossing_t')     # n.s.
kw_comparison(df_words, 'autocorr_t')        # n.s.
kw_comparison(df_words, 'skew_t')            # n.s.

# EUCLIDEAN FASTTEXT
kw_comparison(df_words, 'sum_distances_n')  # p-value: 5.920208000245156e-07

# INTO A TABLE
variables = ['mean_semsim', 'max_semsim', 'min_semsim',
             'av_ssc', 'av_crossing', 'autocorr',
             'mean_semsim_t', 'max_semsim_t', 'min_semsim_t',
             'av_ssc_t', 'av_crossing_t', 'autocorr_t']

#kw_test = pd.DataFrame(columns=['Variable', 'H-statistic', 'p-value'])
kw_test = []
for var in variables:
    groups = df_words['group'].unique()
    group_data = [
        df_words[df_words['group'] == group][var].dropna()  # Filter out NaN values
        for group in groups
    ]
    statistic, p_value = kruskal(*group_data)
    #kw_test = kw_test.append({'Variable': var, 'H-statistic': statistic, 'p-value': p_value}, ignore_index=True)
    kw_test.append({'Variable': var, 'H-statistic': statistic, 'p-value': p_value})

kw_test = pd.DataFrame(kw_test)
kw_test.to_excel(path_tables+'table_4.xlsx', index=False)

