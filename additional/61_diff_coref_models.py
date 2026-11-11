# Additional analysis; check additional/README.md before running.
from pipeline.paths import METADATA, ADDITIONAL_RESULTS

"""
@author: Claudio
"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# PATH INPUT
path_demo = METADATA
path_results = ADDITIONAL_RESULTS


""" FUNCTIONS """

# COMPARE COREF MODELS
def plot_and_compare(df, column1, column2, name1, name2, group=False):
    # Extract columns based on group identifier
    if group==False:
        column1 = df[column1]
        column2 = df[column2]
    else:
        column1 = df[df['group'] == group][column1]
        column2 = df[df['group'] == group][column2]
    # Perform t-test
    t_statistic, p_value = ttest_ind(column1, column2, nan_policy='omit')
    # Print t-statistic and p-value
    print(f"t-statistic: {t_statistic}")
    print(f"p-value: {p_value}")
    # Plot distribution using violin plot
    sns.violinplot(data=[column1, column2])
    plt.xlabel('Models')
    plt.ylabel('CONLL score')
    if group==False:
        plt.title('Distribution of CONLL scores - All groups')
    else:
        plt.title(f'Distribution of CONLL scores - {group}')
    plt.xticks(ticks=[0, 1], labels=[f'{name1}', f'{name2}'])
    plt.show()
    """
    if group==False:
        plt.savefig(path_results+'conll_violin_plot_all.jpg', dpi=500)
    else:
        plt.savefig(path_results+f'conll_violin_plot_{group}.jpg', dpi=500)
    """
    

# COMPARE CONLL SCORES BETWEEN GROUPS
def compare_between_groups(df, column_name):
    # Extract unique groups
    groups = df['group'].unique()
    # Perform t-tests between each pair of groups
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            group1 = groups[i]
            group2 = groups[j]
            # Extract values for the specified column and groups
            values_group1 = df[df['group'] == group1][column_name]
            values_group2 = df[df['group'] == group2][column_name]
            # Perform t-test, ignoring NaN values
            t_statistic, p_value = ttest_ind(values_group1, values_group2, nan_policy='omit')
            # Print t-statistic and p-value
            print(f"t-statistic for {column_name} between {group1} and {group2}: {t_statistic}")
            print(f"p-value for {column_name} between {group1} and {group2}: {p_value}")
            # Plot distribution using violin plot
            sns.violinplot(data=[values_group1, values_group2])
            plt.xlabel('Groups')
            plt.ylabel('CONLL score')
            plt.title(f'Differences of {column_name} between {group1} and {group2}')
            plt.xticks(ticks=[0, 1], labels=[group1, group2])
            #plt.show()
            plt.savefig(path_results+f'{column_name}_violin_plot_{group1}_{group2}.jpg', dpi=500)
            plt.close()


# SCATTER GROUPS CORR
def scatter_plot_group(df, var1, var2, group=None, pict=None, filename='', use_linear=True):
    # Filter the DataFrame based on 'group' and 'pict'
    if group is not None:
        df = df[df['group'] == group]
    if pict is not None:
        df = df[df['bild'] == pict]
    # Remove NaN
    df = df.dropna(subset=[var1, var2])
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
    x_axis = f'{var1}'
    y_axis = f'{var2}'
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'{y_axis} vs {x_axis} - {group}')
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
    #plt.show()
    return corr, p_value



""" ANALYZE """
    
# IMPORT ALL DATA
df = pd.read_csv(path_demo+'ssd_mdd_hc_data.csv')

# KEEP RESULTS MODELS AND GROUPS
df = df[['Proband', 'Typ2', 'SANS_Sum_ENIGMA', 
         'SAPS_PositiveFormaleDenkstoerung_ENIGMA',
         'SAPS_Sum_ENIGMA', 'IQ',
         'CONLL_AV_TuBA', 'CONLL_AV_C2F_DROC']]

# REPLACE -99 AS NAN
df.replace([-99, -93], np.nan, inplace=True)

# RENAME
df.rename(columns={'Typ2': 'group', 'SANS_Sum_ENIGMA': 'SANS_Sum',
                   'SAPS_PositiveFormaleDenkstoerung_ENIGMA': 'SAPS_FTD',
                   'SAPS_Sum_ENIGMA': 'SAPS_Sum',
                   'CONLL_AV_C2F_DROC': 'CONLL_AV_C2F'}, inplace=True)

# COMPARE MODELS
plot_and_compare(df, 'CONLL_AV_TuBA', 'CONLL_AV_C2F', 'TuBA', 'C2F')
plot_and_compare(df, 'CONLL_AV_TuBA', 'CONLL_AV_C2F', 'TuBA', 'C2F', group='HC')
plot_and_compare(df, 'CONLL_AV_TuBA', 'CONLL_AV_C2F', 'TuBA', 'C2F', group='SSD')
plot_and_compare(df, 'CONLL_AV_TuBA', 'CONLL_AV_C2F', 'TuBA', 'C2F', group='MDD')

# COMPARE GROUPS
compare_between_groups(df, 'CONLL_AV_TuBA')
compare_between_groups(df, 'CONLL_AV_C2F')

# SCATTER CORRELATIONS

# SANS SUM
scatter_plot_group(df, 'SANS_Sum', 'CONLL_AV_TuBA',  group='MDD', pict=None, filename='SANS_TUBA_MDD')
scatter_plot_group(df, 'SANS_Sum', 'CONLL_AV_TuBA',  group='SSD', pict=None, filename='SANS_TUBA_SSD')

# SAPS SUM
scatter_plot_group(df, 'SAPS_Sum', 'CONLL_AV_TuBA',  group='MDD', pict=None, filename='SAPS_TUBA_MDD')
scatter_plot_group(df, 'SAPS_Sum', 'CONLL_AV_TuBA',  group='SSD', pict=None, filename='SAPS_TUBA_SSD')

# SAPS FTD
scatter_plot_group(df, 'SAPS_FTD', 'CONLL_AV_TuBA',  group='MDD', pict=None, filename='FTD_TUBA_MDD')
scatter_plot_group(df, 'SAPS_FTD', 'CONLL_AV_TuBA',  group='SSD', pict=None, filename='FTD_TUBA_SSD')

