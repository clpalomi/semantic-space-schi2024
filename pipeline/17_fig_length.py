# Length boxplots from the supplied script, moved after the analyses.
from pipeline.paths import TEXTS, METADATA, RESULTS

# -*- coding: utf-8 -*-
"""
@author: Claudio
"""


# IMPORT PACKAGES
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import itertools

# PATH INPUT
path_input = TEXTS
path_demo = METADATA
path_results = RESULTS

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
            if col == 'total_length':
                title = 'Total length'
            elif col == 'sent_av_length':
                title = 'Average sentence length'
            else:
                title = col
            plt.subplot(1, len(numerical_cols), i)
            # Create a boxplot for the current 'bild' with different colors for each 'group'
            ax = sns.boxplot(x=hue_col, y=col, data=subset_df)
            # Set plot title
            plt.title(f'{title} - Picture {bild_value} - by {hue_col}', fontsize=14)
            # Set x-axis and y-axis label size
            ax.set_xlabel(hue_col.capitalize(), fontsize=13)
            ax.set_ylabel(title.capitalize(), fontsize=13)
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
            # corrected_p_values = multipletests(p_values, method='bonferroni')[1]
            corrected_p_values = multipletests(p_values, method='bonferroni')[1]
            # Use Annotator to show significant differences
            annotator = Annotator(ax, group_pairs, data=subset_df, x=hue_col, y=col)
            annotator.set_pvalues(corrected_p_values)
            annotator.annotate()
        plt.tight_layout()
        plt.savefig(path_results+f'length_stats_picture{bild_value}.jpg', dpi=1000)
        plt.show()


stats_length = pd.read_pickle(path_results+'df_stats_length.pkl')
boxplot_comparison(stats_length, numerical_cols=['total_length', 'sent_av_length'])
