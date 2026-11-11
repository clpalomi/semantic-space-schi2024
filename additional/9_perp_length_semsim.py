# Additional analysis; check additional/README.md before running.
from pipeline.paths import ADDITIONAL_RESULTS

# -*- coding: utf-8 -*-
"""
@author: Claudio
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# PATHS

path_results = ADDITIONAL_RESULTS



""" FUNCTIONS """

# SCATTER PLOT GROUPS
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
    # Calculate correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(df[var1], df[var2])
    # Add labels and title
    plt.xlabel(f'{var1}')
    plt.ylabel(f'{var2}')
    plt.title(f'{var2} vs {var1}')
    # plot line
    if p_value < 0.05:
        sns.regplot(x=var1, y=var2, data=df, scatter=False, color='red')
        plt.legend()
    #plt.savefig(path_results+'scatter_corr.jpg', dpi=500)
    plt.show()
    return [correlation_coefficient, p_value]
   

# BOXPLOT DIFFERENCES BETWEEN GROUPS
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



""" READ DATASETS """

# semsim length
df_sem = pd.read_pickle(path_results+'df_semsim_dist_length.pkl')
# perplexity
# df_per = pd.read_pickle(path_results+'df_perplexity_all.pkl')
df_per = pd.read_pickle(path_results+'df_perplexity_type_all.pkl')

# re-order
column_order = ['name', 'group', 'bild'] + [col for col in df_per.columns if col not in ['name', 'group', 'bild', 's_group', 'type']]
df_per = df_per[column_order]
# Drop columns 's_group' and 'type'
to_drop = ['s_group', 'type']
df_per = df_per.drop(columns=to_drop, errors='ignore')
# Rename columns
column_mapping = {
    'mean': 'mean_per',
    'std': 'std_per',
    'range_s': 'range_per',
    'per10': 'per10_per',
    'per90': 'per90_per',
}
df_per = df_per.rename(columns=column_mapping)

# MERGE ALL
df = pd.merge(df_per, df_sem, on=['name', 'group', 'bild'])



""" SCATTER CORRELATIONS """
    
# ALL PERPLEXITY
scatter_plot_group(df, 'n_pron', 'mean_semsim')       # 0.48
scatter_plot_group(df, 'n_det', 'mean_semsim')        # -0.56
scatter_plot_group(df, 'n_words', 'mean_semsim')      # NO-CORR
scatter_plot_group(df, 'n_content_y', 'mean_semsim')  # NO-CORR


# AV PERPLEXITY
scatter_plot_group(df, 'mean_per', 'mean_semsim')      # 0.15
scatter_plot_group(df, 'mean_per', 'total_length')     # 0.18
scatter_plot_group(df, 'mean_per', 'sum_distances')    # 0.23
scatter_plot_group(df, 'mean_per', 'sum_distances_n')  # 0.31
scatter_plot_group(df, 'mean_per', 'slope')            # NO-CORR
scatter_plot_group(df, 'mean_per', 'max_semsim')       # 0.13
scatter_plot_group(df, 'mean_per', 'av_ssc')           # NO-CORR
scatter_plot_group(df, 'mean_per', 'skew')             # NO-CORR
scatter_plot_group(df, 'mean_per', 'av_crossing')      # - 0.09
scatter_plot_group(df, 'mean_per', 'av_crossing_t')    # - 0.3
scatter_plot_group(df, 'mean_per', 'av_ssc_t')         # NO-CORR
scatter_plot_group(df, 'mean_per', 'mean_semsim_t')    # NO-CORR



""" BOX-PLOTS """

# ALL PERPLEXITY
boxplot_group_diff(df, 'n_pron')
boxplot_group_diff(df, 'n_noun_phrases')
boxplot_group_diff(df, 'mean_pron')
boxplot_group_diff(df, 'mean_det')
boxplot_group_diff(df, 'mean_content')



# AV PERPLEXITY
boxplot_group_diff(df, 'mean_per')
boxplot_group_diff(df, 'range_per')
boxplot_group_diff(df, 'total_length')
boxplot_group_diff(df, 'mean_semsim')

boxplot_group_diff(df, 'sum_distances')
boxplot_group_diff(df, 'sum_distances_n')   #



""" MIXED-EFFECTS REGRESSION """

from statsmodels.regression.mixed_linear_model import MixedLM

def mixed_model_results(y, x):
    model_r = MixedLM.from_formula(f'{y} ~ {x} + group + bild', df, groups=df["name"])
    result  = model_r.fit()
    print(result.summary())
    
    
mixed_model_results('mean_per', 'total_length')
mixed_model_results('range_per', 'total_length')
mixed_model_results('per90_per', 'total_length')
mixed_model_results('per10_per', 'total_length')

mixed_model_results('mean_semsim', 'n_content')
mixed_model_results('autocorr', 'n_content')
mixed_model_results('mean_semsim_t', 'total_length')

mixed_model_results('sum_distances', 'n_content')
mixed_model_results('sum_distances_n', 'n_content')
    

# ALL PERPLEXITY
mixed_model_results('mean_content', 'n_content_x')
mixed_model_results('mean_pron', 'n_content_x')
    
