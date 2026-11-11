from pipeline.paths import METADATA, RESULTS, TABLES

"""
@author: Claudio

Tables and results used in the article

"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# PATH
path_demo = METADATA
path_tables = TABLES
path_results = RESULTS

""" 3. SEMSIM VS SPEECH LENGTH (FIG. 2) """ 

df = pd.read_pickle(path_results+'df_semsim_dist_length.pkl')

# SCATTER PLOT WITH LINEAR OR LOG FIT
def scatter_plot_group_line(df, var1, var2, group=None, pict=None, filename='', use_linear=True):
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
            plt.scatter(group_data[var1], group_data[var2], label=f'{name}', s=12)
        plt.legend()
    else:
        plt.scatter(df[var1], df[var2])
    # Add labels and title
    x_axis = 'Speech length'
    y_axis = 'Mean semantic similarity'
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    #plt.title(f'{x_axis} vs {y_axis}')
    # Calculate Pearson correlation coefficient and p-value
    corr, p_value = pearsonr(df[var1], df[var2])
    if use_linear:
        # Fit linear regression
        x = df[var1]
        y = df[var2]
        coeffs = np.polyfit(x, y, 1)
        m, b = coeffs
        x_fit = np.linspace(min(x), max(x), 100)
        plt.plot(x_fit, m * x_fit + b, color='black')
    else:
        # Transform one variable with logarithmic function
        x = df[var1]
        y = df[var2]
        x_log = np.log(x)
        coeffs = np.polyfit(x_log, y, 1)
        m, b = coeffs
        x_fit = np.linspace(min(x), max(x), 100)
        plt.plot(x_fit, m * np.log(x_fit) + b, color='black')
    plt.savefig(path_results+f'scatter_corr_{filename}.jpg', dpi=500)
    plt.show()
    plt.close()
    return corr, p_value
    

# SCATTER PLOT WITH REG LINE
def scatter_plot_group(df, var1, var2, group=None, pict=None, filename=''):
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
            plt.scatter(group_data[var1], group_data[var2], label=f'{name}', s=12)
        plt.legend()
    else:
        plt.scatter(df[var1], df[var2])
    # Calculate Pearson correlation coefficient and p-value
    corr, p_value = pearsonr(df[var1], df[var2])
    # Add regression line if the correlation is significant
    if p_value < 0.05:
        sns.regplot(x=var1, y=var2, data=df, scatter=False, color='black')
    # Add labels and title
    x_axis = 'Speech length'
    y_axis = 'Mean semantic similarity'
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    #plt.title(f'{x_axis} vs {y_axis}')
    plt.savefig(path_results+f'scatter_corr_{filename}.jpg', dpi=500)
    plt.show()
    plt.close()
    return corr, p_value


# FIGURE 2
scatter_plot_group(df, 'total_length', 'mean_semsim_t', filename='BERT')
scatter_plot_group(df, 'total_length', 'mean_semsim', filename='FastText')

# SM MEAN SEMSIM AND LENGTH, EACH GROUP (FIGURE S2)
scatter_plot_group(df, 'total_length', 'mean_semsim_t', group='SSD', filename='SSD_BERT')
scatter_plot_group(df, 'n_content', 'mean_semsim', group='SSD', filename='SSD_FastText')

scatter_plot_group(df, 'total_length', 'mean_semsim_t', group='MDD', filename='MDD_BERT')
scatter_plot_group(df, 'n_content', 'mean_semsim', group='MDD', filename='MDD_FastText')

scatter_plot_group(df, 'total_length', 'mean_semsim_t', group='HC', filename='HC_BERT')
scatter_plot_group(df, 'n_content', 'mean_semsim', group='HC', filename='HC_FastText')

# SENTECNCE AND MEAN SEMSIM, DESCRIBED IN PAPER, NO FIGURE
scatter_plot_group(df, 'sent_av_length', 'mean_semsim_t', filename='sentence_length_BERT')

