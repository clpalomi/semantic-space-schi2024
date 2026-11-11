# Additional analysis; check additional/README.md before running.
from pipeline.paths import METADATA, RESULTS, ADDITIONAL_RESULTS

"""
@author: Claudio
"""

# IMPORT PACKAGES
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

# PATH INPUT
path_input = METADATA
path_results = ADDITIONAL_RESULTS


""" FUNCTIONS """

# P-VALUES MATRIX
def calculate_p_values(df):
    df_cols = pd.DataFrame(columns=df.columns)
    pval_matrix = df_cols.transpose().join(df_cols, how='outer')
    
    for r in df.columns:
        for c in df.columns:
            if r != c:
                # Drop NaN values for correlation calculation
                non_na = df[[r, c]].dropna()
                if not non_na.empty and len(non_na) > 1:
                    _, p = pearsonr(non_na[r], non_na[c])
                    pval_matrix.at[r, c] = round(p, 4)
                else:
                    pval_matrix.at[r, c] = float('nan')
            else:
                pval_matrix.at[r, c] = round(0.0, 4)
    return pval_matrix


""" DATA """

# OPEN SYMPTOMS
df_s = pd.read_excel(path_input+'symptoms.xlsx')
df_s = df_s.replace(-99, np.nan)
df_s = df_s.replace(-93, np.nan)

# OPEN SEMSIM VARIABLES
df_c    = pd.read_pickle(RESULTS+'df_semsim_sent_length.pkl')
df_c_1  = df_c[df_c['bild']=='1']
df_c_2  = df_c[df_c['bild']=='2']
df_c_4  = df_c[df_c['bild']=='4']
df_c_6  = df_c[df_c['bild']=='6']
df_c_av = df_c.groupby('name').mean(numeric_only=True).reset_index()

# OPEN VOLUME
df_sents = pd.read_pickle(path_results+'df_sent_convexhull_5D_mono.pkl')
df_sents['log_volume'] = np.log(df_sents['volume'])
df_sents['log_area']   = np.log(df_sents['area'])
df_sents_1  = df_sents[df_sents['bild']=='1']
df_sents_2  = df_sents[df_sents['bild']=='2']
df_sents_4  = df_sents[df_sents['bild']=='4']
df_sents_6  = df_sents[df_sents['bild']=='6']
df_sents_av = df_sents.groupby('name').mean(numeric_only=True).reset_index()

# MERGE
df   = pd.merge(df_s, df_c_av, on='name', how='inner')
df   = pd.merge(df, df_sents_av, on='name', how='inner')
df_1 = pd.merge(df_s, df_c_1, on='name', how='inner')
df_2 = pd.merge(df_s, df_c_2, on='name', how='inner')
df_4 = pd.merge(df_s, df_c_4, on='name', how='inner')
df_6 = pd.merge(df_s, df_c_6, on='name', how='inner')


""" CALCULATE """

list_var =   ['SANS_Sum_ENIGMA', 'SANS_Affektverflachung_ENIGMA',
              'SANS_AlogieParalogie_ENIGMA', 'SANS_AbulieApathie_ENIGMA',
              'SANS_Anhedonie_ENIGMA', 'SANS_Aufmerksamkeit_ENIGMA',
              'SANS_Sum_ohneAufmerksamkeit_ENIGMA', 'SAPS_Halluzinationen_ENIGMA',
              'SAPS_Wahn_ENIGMA', 'SAPS_BizarresVerhalten_ENIGMA',
              'SAPS_PositiveFormaleDenkstoerung_ENIGMA', 'SAPS_Sum_ENIGMA',
              'HAMD_Sum17', 'HAMA_Sum', 'YMRS_Sum', 'GAFscore', 'IQ',
              'total_length', 'sent_av_length', 'dispersion',
              'sum_dist_n',
              'max_semsim', 'mean_semsim', 'min_semsim',
              'ssc', 'autocorr', 'av_crossing', 
              'sum_distances','log_volume', 'log_area']

list_var_1 = ['SANS_Sum_ENIGMA', 'SANS_Affektverflachung_ENIGMA',
              'SANS_AlogieParalogie_ENIGMA', 'SANS_AbulieApathie_ENIGMA',
              'SANS_Anhedonie_ENIGMA', 'SANS_Aufmerksamkeit_ENIGMA',
              'SANS_Sum_ohneAufmerksamkeit_ENIGMA', 'SAPS_Halluzinationen_ENIGMA',
              'SAPS_Wahn_ENIGMA', 'SAPS_BizarresVerhalten_ENIGMA',
              'SAPS_PositiveFormaleDenkstoerung_ENIGMA', 'SAPS_Sum_ENIGMA',
              'HAMD_Sum17', 'HAMA_Sum', 'YMRS_Sum', 'GAFscore']

list_var_2 = ['total_length', 'sent_av_length', 'dispersion',
              'sum_dist_n',
              'max_semsim', 'mean_semsim', 'min_semsim',
              'ssc', 'autocorr', 'av_crossing', 'log_volume']


df   = df[list_var]
df_1 = df_1[list_var]
df_2 = df_2[list_var]
df_4 = df_4[list_var]
df_6 = df_6[list_var]


# CORRELATIONS
corr_matrix   = df.corr()
corr_matrix_1 = df_1.corr()
corr_matrix_2 = df_2.corr()
corr_matrix_4 = df_4.corr()
corr_matrix_6 = df_6.corr()


def corr_plot(df, corr_matrix, list_var_1, list_var_2, path_results, bild):
    # Calculate P-values
    p_values_matrix = calculate_p_values(df)
    
    # Find significant comparisons
    significant_comparisons = []
    for row in list_var_1:
        for col in list_var_2:
            if row in p_values_matrix.index and col in p_values_matrix.columns:
                if pd.notna(p_values_matrix.loc[row, col]) and p_values_matrix.loc[row, col] < 0.05:
                    correlation = corr_matrix.loc[row, col]
                    significant_comparisons.append((row, col, correlation, p_values_matrix.loc[row, col]))

    for col in list_var_1:
        for row in list_var_2:
            if col in p_values_matrix.index and row in p_values_matrix.columns:
                if pd.notna(p_values_matrix.loc[row, col]) and p_values_matrix.loc[row, col] < 0.05:
                    correlation = corr_matrix.loc[row, col]
                    significant_comparisons.append((col, row, correlation, p_values_matrix.loc[row, col]))

    # Sort by the absolute value of the correlation coefficient
    significant_comparisons = sorted(significant_comparisons, key=lambda x: abs(x[2]), reverse=True)
    significant_comparisons = set(significant_comparisons)
    significant_comparisons = list(significant_comparisons)
    significant_comparisons = sorted(significant_comparisons, key=lambda x: abs(x[2]), reverse=True)
    
    # Create the correlation matrix for the heatmap
    correlation_values = [comparison[2] for comparison in significant_comparisons]
    num_var_1 = len(list_var_1)
    num_var_2 = len(list_var_2)
    correlation_matrix = np.zeros((num_var_1, num_var_2))

    for comparison in significant_comparisons:
        var_1_index = list_var_1.index(comparison[0])
        var_2_index = list_var_2.index(comparison[1])
        correlation_matrix[var_1_index, var_2_index] = comparison[2]

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", 
                xticklabels=list_var_2, yticklabels=list_var_1, linewidths=0.5)
    plt.title(f'Correlation Coefficients - {bild}', fontsize=16)
    plt.xlabel('Semsim variables', fontsize=14)
    plt.ylabel('Symptoms', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f'{path_results}heatmap_correlation_semsim_symptoms_{bild}.png', dpi=500, bbox_inches='tight')
    plt.show()
    
    return significant_comparisons


# PLOT
significant_comparisons   = corr_plot(df, corr_matrix, list_var_1, list_var_2, path_results, 'all')
significant_comparisons_1 = corr_plot(df_1, corr_matrix_1, list_var_1, list_var_2, path_results, '1')
significant_comparisons_2 = corr_plot(df_2, corr_matrix_2, list_var_1, list_var_2, path_results, '2')
significant_comparisons_4 = corr_plot(df_4, corr_matrix_4, list_var_1, list_var_2, path_results, '4')
significant_comparisons_6 = corr_plot(df_6, corr_matrix_6, list_var_1, list_var_2, path_results, '6')



