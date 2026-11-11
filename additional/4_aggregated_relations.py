# Additional analysis; check additional/README.md before running.
from pipeline.paths import METADATA, ADDITIONAL_RESULTS

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 10:55:49 2023
@author: Claudio
"""

# IMPORT PACKAGES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
from scipy import stats

# PATH INPUT
path_input = METADATA
path_results = ADDITIONAL_RESULTS


""" FUNCTIONS """

def graph_scatter(df, var1, var2):
    corr_coeffs = []
    p_values    = []
    groups = df['group'].unique()
    # Create a color map for groups
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))
    # Plot scatter points for each group
    for i, group in enumerate(groups):
        group_df = df[df['group'] == group]
        plt.scatter(group_df[var1], group_df[var2], color=colors[i], label=group)
        # Calculate correlation coefficient and p-value
        corr_coeff, p_value = pearsonr(group_df[var1], group_df[var2])
        corr_coeffs.append(corr_coeff)
        p_values.append(p_value)
        print(f'Group: {group}')
        print(f'Correlation coefficient: {corr_coeff:.2f}')
        print(f'P-value: {p_value:.4f}')
        # Perform Fisher's z-test
    # Add labels and title
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title('Scatter Plot of '+ var1 +' vs '+ var2)
    plt.legend()
    plt.show()
    return [corr_coeffs, p_values]


#  BOX-PLOT WIHT TTEST (SEABORN)
def plot_with_ttest_2(df, x, y, path=False, name_file=False):
    # GRAPHS AND DIFFERENCES
    sns.boxplot(x=x, y=y, data=df, showfliers=False)
    # Perform t-test
    group1 = df[df[x] == df[x].unique()[0]][y]
    group2 = df[df[x] == df[x].unique()[1]][y]
    # Handle NaN values
    group1 = group1.dropna()
    group2 = group2.dropna()
    t_statistic, p_value = stats.ttest_ind(group1, group2)
    # Add significance indicator
    if p_value < 0.05:
        significance_indicator = '*'
    else:
        significance_indicator = 'ns'
    # Add p-value to the plot
    plt.text(0.5, 1.05, f'p-value: {p_value:.4f} {significance_indicator}',
             horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes)
    # Increase the size of the axis titles
    # Enlarge x and y labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(x, fontdict={'fontsize': 14})
    plt.ylabel(y, fontdict={'fontsize': 14})
    if path!=False:
        plt.savefig(path_results+name_file+'.eps', dpi=1200, bbox_inches='tight')
        plt.savefig(path_results+name_file+'.jpg', dpi=1200, bbox_inches='tight')
    plt.show()
    


""" PROCESS """

# READ FILES
df_ent = pd.read_pickle(ADDITIONAL_RESULTS+'df_perplexity_all.pkl')
del df_ent['group']
del df_ent['type']
var_c = {'mean': 'mean_ent', 'std': 'std_ent',
           'range_s': 'range_s_ent', 'per10': 'per10_ent', 'per90': 'per90_ent'}
df_ent.rename(columns=var_c, inplace=True)
del df_ent['s_group']

df_sem = pd.read_pickle(path_results+'df_semsim_german.pkl')
df = pd.merge(df_ent, df_sem, on=['name', 'bild'], how='inner')
var_c = {'mean': 'mean_sem', 'std': 'std_sem',
           'range_s': 'range_s_sem', 'per10': 'per10_sem', 'per90': 'per90_sem'}
df.rename(columns=var_c, inplace=True)
del df['type']

# CHANGE ORDER
columns = df.columns.tolist()
order   = ['name', 'group', 's_group', 'bild']
columns = [col for col in columns if col not in order]
df = df[order + columns]

# MERGE WITH SYMPTOMS
df_s = pd.read_excel(path_input+'symptoms.xlsx')
df_s = df_s.replace(-99, np.nan)
df_s = df_s.replace(-93, np.nan)
df = pd.merge(df, df_s, on='name', how='inner')


# missing data

# DROP IF TOO MANY NAN VALUES
n_obs1 = len(df)   # 477 observaciones
nan_counts = df.isna().sum(axis=1)
df = df[nan_counts < 8]
n_obs2 = len(df)   # 444

# Calculate the average of each column by 'group' (ignoring NaN values)
group_means = df.groupby('group').transform(lambda x: x.fillna(x.mean()))

# EXCLUDE COLUMNS FROM REPLACE OF NAN
excluded = ['name', 'group', 's_group', 'bild', 'mean_ent', 'std_ent',
            'range_s_ent', 'per10_ent', 'per90_ent', 'mean_sem', 'std_sem',
            'range_s_sem', 'per10_sem', 'per90_sem', 'age', 'sex']

# Loop through columns and replace NaN values
for col in df.columns:
    if col not in excluded:
        df[col].fillna(group_means[col], inplace=True)

# RESET INDEX        
df = df.reset_index(drop=True)

"""
# VALUES FOR EACH TASK
grouped = df.groupby('bild')
grouped_dfs = {}
for name, group in grouped:
    grouped_dfs[name] = group.copy()
df_1 = grouped_dfs['1']
df_2 = grouped_dfs['2']
df_4 = grouped_dfs['4']
df_6 = grouped_dfs['6']
df_1 = df_1.iloc[:, 4:]
df_2 = df_2.iloc[:, 4:]
df_4 = df_4.iloc[:, 4:]
df_6 = df_6.iloc[:, 4:]
"""


# ONLY SZ
df_sz = df.loc[df['group'] == 'Sz']
df_sz = df_sz.iloc[:, 4:]
to_drop = ['range_s_ent', 'per10_ent', 'per90_ent', 'per10_sem', 'per90_sem']
df_sz = df_sz.drop(columns=to_drop)

# ONLY MD
df_md = df.loc[df['group'] == 'MD']
df_md = df_md.iloc[:, 4:]
to_drop = ['range_s_ent', 'per10_ent', 'per90_ent', 'per10_sem', 'per90_sem']
df_md = df_md.drop(columns=to_drop)

# ONLY HC
df_hc = df.loc[df['group'] == 'HC']
df_hc = df_hc.iloc[:, 4:]
to_drop = ['range_s_ent', 'per10_ent', 'per90_ent', 'per10_sem', 'per90_sem']
df_hc = df_hc.drop(columns=to_drop)

# ALL
df   = df.iloc[:, 4:]
to_drop = ['range_s_ent', 'per10_ent', 'per90_ent', 'per10_sem', 'per90_sem']
df = df.drop(columns=to_drop)


""" CORRELATIONS """


# ALL
corr_matrix = df.corr(numeric_only=True)
p_values = np.zeros(corr_matrix.shape)
for i in range(corr_matrix.shape[0]):
    for j in range(i+1, corr_matrix.shape[1]):
        # Calculate Pearson correlation coefficient and p-value
        corr, p_val = pearsonr(df.iloc[:, i], df.iloc[:, j])
        p_values[i, j] = p_val
        p_values[j, i] = p_val
p_values_df = pd.DataFrame(p_values, index=df.columns, columns=df.columns)
# TO EXCEL
corr_matrix = corr_matrix.iloc[:, :7]
p_values_df = p_values_df.iloc[:, :7]
df_corr = pd.concat([corr_matrix, p_values_df], axis=1)
df_corr.to_excel(path_results+'correlations_symptoms.xlsx', index=True)


# SZ
corr_matrix_sz = df_sz.corr(numeric_only=True)
p_values = np.zeros(corr_matrix_sz.shape)
for i in range(corr_matrix_sz.shape[0]):
    for j in range(i+1, corr_matrix_sz.shape[1]):
        # Calculate Pearson correlation coefficient and p-value
        corr, p_val = pearsonr(df_sz.iloc[:, i], df_sz.iloc[:, j])
        p_values[i, j] = p_val
        p_values[j, i] = p_val
p_values_df_sz = pd.DataFrame(p_values, index=df_sz.columns, columns=df_sz.columns)
# TO EXCEL
corr_matrix_sz = corr_matrix_sz.iloc[:, :7]
p_values_df_sz = p_values_df_sz.iloc[:, :7]
df_corr_sz = pd.concat([corr_matrix_sz, p_values_df_sz], axis=1)
df_corr_sz.to_excel(path_results+'correlations_symptoms_sz.xlsx', index=True)


# MD
corr_matrix_md = df_md.corr(numeric_only=True)
p_values = np.zeros(corr_matrix_md.shape)
for i in range(corr_matrix_md.shape[0]):
    for j in range(i+1, corr_matrix_md.shape[1]):
        # Calculate Pearson correlation coefficient and p-value
        corr, p_val = pearsonr(df_md.iloc[:, i], df_md.iloc[:, j])
        p_values[i, j] = p_val
        p_values[j, i] = p_val
p_values_df_md = pd.DataFrame(p_values, index=df_md.columns, columns=df_md.columns)
# TO EXCEL
corr_matrix_md = corr_matrix_md.iloc[:, :7]
p_values_df_md = p_values_df_md.iloc[:, :7]
df_corr_md = pd.concat([corr_matrix_md, p_values_df_md], axis=1)
df_corr_md.to_excel(path_results+'correlations_symptoms_md.xlsx', index=True)


# HC
corr_matrix_hc = df_hc.corr(numeric_only=True)
p_values = np.zeros(corr_matrix_hc.shape)
for i in range(corr_matrix_hc.shape[0]):
    for j in range(i+1, corr_matrix_hc.shape[1]):
        # Calculate Pearson correlation coefficient and p-value
        corr, p_val = pearsonr(df_hc.iloc[:, i], df_hc.iloc[:, j])
        p_values[i, j] = p_val
        p_values[j, i] = p_val
p_values_df_hc = pd.DataFrame(p_values, index=df_hc.columns, columns=df_hc.columns)
# TO EXCEL
corr_matrix_hc = corr_matrix_hc.iloc[:, :7]
corr_matrix_hc = corr_matrix_hc.dropna(how='all')
p_values_df_hc = p_values_df_hc.iloc[:, :7]
p_values_df_hc = p_values_df_hc.dropna(how='all')
df_corr_hc = pd.concat([corr_matrix_hc, p_values_df_hc], axis=1)
df_corr_hc.to_excel(path_results+'correlations_symptoms_hc.xlsx', index=True)





""" GRAPH DIFFERENCES """

graph_scatter(df, 'mean_sem', 'std_sem')
graph_scatter(df, 'mean_sem', 'range_s_sem')
graph_scatter(df, 'mean_sem', 'SANS_Sum_ENIGMA')
graph_scatter(df, 'mean_sem', 'SANS_Affektverflachung_ENIGMA')
graph_scatter(df, 'mean_sem', 'SANS_AlogieParalogie_ENIGMA')
graph_scatter(df, 'mean_sem', 'SANS_Sum_ohneAufmerksamkeit_ENIGMA')
graph_scatter(df, 'mean_sem', 'SAPS_PositiveFormaleDenkstoerung_ENIGMA')
graph_scatter(df, 'mean_sem', 'GAFscore')



plot_with_ttest_2(df, 'group', 'mean_sem', True, 'mean_semsim_all')




""" MLE """
from statsmodels.regression.mixed_linear_model import MixedLM

model = MixedLM.from_formula('mean_sem  ~ group + bild + IQ + ', df, groups=df["name"])
result  = model.fit()
result.summary()

