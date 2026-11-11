# Additional analysis; check additional/README.md before running.
from pipeline.paths import METADATA, ADDITIONAL_RESULTS

"""
@author: Claudio
"""


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM


""" PATHS """
path_demo = METADATA
path_results = ADDITIONAL_RESULTS


""" FUNCTIONS """


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
    # Perform t-test comparisons between all groups
    groups = df['group'].unique()
    p_values = {}
    for i in range(len(groups) - 1):
        for j in range(i + 1, len(groups)):
            group1 = groups[i]
            group2 = groups[j]
            subset1 = df[df['group'] == group1][var1]
            subset2 = df[df['group'] == group2][var1]
            t_stat, p_value = stats.ttest_ind(subset1, subset2)
            key = f'{group1} vs {group2}'
            p_values[key] = p_value
    # Show the plot
    plt.show()
    return p_values


# SCATTER
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


# MIXED EFFECTS MODEL
def mixed_model_results(df, y, x):
    model_r = MixedLM.from_formula(f'{y} ~ group + {x} + bild', df, groups=df["name"])
    result  = model_r.fit()
    print(result.summary())
    return result


# MIXED EFFECTS MODEL
def mixed_model_results_2(df, y, x1, x2):
    model_r = MixedLM.from_formula(f'{y} ~ group + {x1} + {x2} + bild', df, groups=df["name"])
    result  = model_r.fit()
    print(result.summary())
    return result
    


""" VARIABLE DATABASES """

# RECURRENCES
df_ent = pd.read_pickle(path_results+'df_recurrences.pkl')

# SEMSIM
df_semsim = pd.read_pickle(path_results+'df_semsim_dist_length.pkl')
df_semsim['dispersion'] = df_semsim['dispersion'].astype('float64')
df_semsim['sum_distances_2'] = df_semsim['sum_distances_2'].astype('float64')


# REC + SEMSIM
df = pd.merge(df_ent, df_semsim, on=['name', 'group', 'bild'], how='inner')


""" CORR - SCATTERPLOTS """

# RECURRENCES AND SEMSIM
graph_scatter(df, 'tot_rec_n', 'mean_semsim')
graph_scatter(df, 'sum_distances', 'mean_semsim')
graph_scatter(df, 'sum_distances_n', 'mean_semsim_t')
graph_scatter(df, 'sum_distances', 'tot_rec_n')
graph_scatter(df, 'sum_distances_n', 'tot_rec_n')




""" ANALYSIS """

# RECURRENCES
boxplot_group_diff(df_ent, 'num_ent')     # NS
boxplot_group_diff(df_ent, 'n_words')     # HC > SZ / HC > MD
boxplot_group_diff(df_ent, 'tot_rec')     # HC > SZ / HC > MD   pvalue < 0.1
boxplot_group_diff(df_ent, 'av_rec')      # HC > SZ (pvalue < 0.1)
boxplot_group_diff(df_ent, 'tot_rec_n')   # SZ > HC (pvalue < 0.05)

# SEMSIM
boxplot_group_diff(df, 'sum_distances_n')   # SZ > MD > HC 
boxplot_group_diff(df, 'dispersion')        # NS
boxplot_group_diff(df, 'mean_semsim')       # NS


""" MIXED MODEL """
    
# RECURRENCES 
r1 = mixed_model_results(df_ent, 'tot_rec', 'n_words')   # SZ > HC, picts, n_words
r2 = mixed_model_results(df_ent, 'num_ent', 'n_words')   # picts, n_words 
r3 = mixed_model_results(df_ent, 'av_rec', 'n_words')    # picts, n_words 

# SEMSIM
r4 = mixed_model_results(df, 'mean_semsim', 'n_content')    # picts, n_words 
r5 = mixed_model_results_2(df, 'mean_semsim', 'n_content', 'tot_rec_n')

r6 = mixed_model_results_2(df, 'mean_semsim_t', 'total_length', 'tot_rec')   




# FASTTEXT

mixed_model_results(df, 'sum_distances', 'total_length')
mixed_model_results_2(df, 'sum_distances', 'total_length', 'tot_rec_n')

mixed_model_results(df, 'sum_distances_n', 'total_length')
mixed_model_results_2(df, 'sum_distances_n', 'total_length', 'tot_rec_n')



""" PRINT GRAPHS """

def get_summary_mixed(df, var, x, name='sig'):
    model = MixedLM.from_formula(f'{var} ~ group + {x} + bild', df, groups=df["name"])
    result  = model.fit()
    
    # RESULTS TO DATAFRAME
    p_values   = result.pvalues
    coeff      = result.params
    results_df = pd.DataFrame({'Coefficient': coeff, 'P-Value': p_values})
    results_df.to_excel(path_results+f'{name}_{var}.xlsx', index=True)
    
    # PLOT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    color_palette = sns.color_palette("colorblind")
    sns.boxplot(x='group', y=f'{var}', data=df, palette=color_palette, ax=ax1)
    ax1.set_title(f'Box Plot of {var} by Group', fontsize=17)
    ax1.set_xlabel('Group', fontsize=15)  
    ax1.set_ylabel(f'{var}', fontsize=15)  
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    sns.pointplot(x='bild', y=f'{var}', hue='group', data=df, errorbar=None, 
                  markers=['s', 'o', 'v'], palette=color_palette, 
                  ax=ax2)
    ax2.set_xlabel('Task', fontsize=15)
    ax2.set_ylabel(f'{var}', fontsize=15)
    ax2.legend(fontsize=15)
    ax2.set_title(f'Interaction Plot of Group and Picture on {var}', fontsize=17)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    plt.savefig(path_results+f'{name}_{var}.jpg', dpi=1200, bbox_inches='tight')
    #plt.savefig(path_results+f'{name}_{var}.eps', dpi=1200, bbox_inches='tight')
    plt.show()
    return result.summary()


get_summary_mixed(df, 'mean_semsim', 'n_content')

get_summary_mixed(df, 'p25_diff', 'n_content')

get_summary_mixed(df, 'mean_semsim_t', 'total_length')
get_summary_mixed(df, 'max_semsim_t', 'total_length')


# FASTTTEXT
get_summary_mixed(df, 'sum_distances', 'n_content')
get_summary_mixed(df, 'sum_distances_n', 'n_content')


# BERT
get_summary_mixed(df, 'max_semsim_t', 'total_length')
get_summary_mixed(df, 'av_crossing_t', 'total_length')
get_summary_mixed(df, 'av_ssc_t', 'total_length')
get_summary_mixed(df, 'autocorr_t', 'total_length')






