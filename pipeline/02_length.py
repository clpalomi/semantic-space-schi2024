from pipeline.text_measures import read_files_sent, sentence_lengths, length_statistics

from pipeline.paths import TEXTS, METADATA, RESULTS

# -*- coding: utf-8 -*-
"""
@author: Claudio
"""


# IMPORT PACKAGES
import pandas as pd
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from scipy.stats import ttest_ind

# PATH INPUT
path_input = TEXTS
path_demo = METADATA
path_results = RESULTS


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
stats_length.to_pickle(path_results+'df_stats_length.pkl')

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
        p_values = []
        for group1, group2 in group_combinations:
            data_group1 = bild_data[bild_data['group'] == group1][variable]
            data_group2 = bild_data[bild_data['group'] == group2][variable]
            # Perform t-test
            _, p_value = ttest_ind(data_group1, data_group2)
            p_values.append(p_value)
        # Perform Bonferroni correction
        corrected_p_values = multipletests(p_values, method='bonferroni')[1]
        # Append results to the significance dataframe
        for i, ((group1, group2), p_value_corrected) in enumerate(zip(group_combinations, corrected_p_values)):
            result_dict = {
                'bild': bild,
                'variable': variable,
                'group1': group1,
                'group2': group2,
                'p_value': p_values[i],
                'p_value_corrected': p_value_corrected
            }
            significance_data.append(result_dict)

significance_df = pd.concat([pd.DataFrame([result]) for result in significance_data], ignore_index=True)
significance_df = significance_df[['variable', 'bild', 'group1', 'group2',
                                   'p_value', 'p_value_corrected']]
significance_df.to_excel(path_results+'pvalue_stats_length.xlsx', index=False)
significance_df.to_pickle(path_results+'df_significance.pkl')

