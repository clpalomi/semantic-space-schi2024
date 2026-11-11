# Use the 5D monolingual result; see the encoding notes in docs/REPRODUCIBILITY.md.
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

""" 8. CORRELATION WITH SYMPTOMS (FIGURE 5 & FIGURE S3) """
from sklearn.linear_model import LinearRegression

path_input = METADATA


""" DATA """

# OPEN SYMPTOMS
df_sy = pd.read_excel(path_input+'symptoms.xlsx')
df_sy = df_sy.replace(-99, np.nan)
df_sy = df_sy.replace(-93, np.nan)

# OPEN SEMSIM VARIABLES WORDS
df_w   = pd.read_pickle(RESULTS+'df_semsim_dist_length.pkl')
df_w_1  = df_w[df_w['bild']=='1']
df_w_2  = df_w[df_w['bild']=='2']
df_w_4  = df_w[df_w['bild']=='4']
df_w_6  = df_w[df_w['bild']=='6']
df_w_av = df_w.groupby('name').mean(numeric_only=True).reset_index()

# OPEN SEMSIM VARIABLES SENTENCES
df_s   = pd.read_pickle(RESULTS+'df_semsim_sent_length.pkl')
df_s = df_s[['name', 'group', 'bild', 'dispersion']]
column_names = {'dispersion': 'dispersion_sents'}
df_s = df_s.rename(columns=column_names)
df_s_1  = df_s[df_s['bild']=='1']
df_s_2  = df_s[df_s['bild']=='2']
df_s_4  = df_s[df_s['bild']=='4']
df_s_6  = df_s[df_s['bild']=='6']
df_s_av = df_s.groupby('name').mean(numeric_only=True).reset_index()


# OPEN VOLUME
df_v = pd.read_pickle(path_results+'df_sent_convexhull_5D_mono.pkl')
df_v['log_volume'] = np.log(df_v['volume'])
df_v['log_area']   = np.log(df_v['area'])
df_v_1  = df_v[df_v['bild']=='1']
df_v_2  = df_v[df_v['bild']=='2']
df_v_4  = df_v[df_v['bild']=='4']
df_v_6  = df_v[df_v['bild']=='6']
df_v_av = df_v.groupby('name').mean(numeric_only=True).reset_index()
df_v_av = pd.merge(df_v_av, df_v[['name', 'group']].drop_duplicates(), on='name', how='left')


# MERGE
df   = pd.merge(df_sy, df_w_av, on='name', how='inner')
df   = pd.merge(df, df_s_av, on='name', how='inner')
df   = pd.merge(df, df_v_av, on='name', how='inner')
df_1 = pd.merge(df_sy, df_w_1, on='name', how='inner')
df_1 = pd.merge(df_1, df_s_1, on='name', how='inner')
df_1 = pd.merge(df_1, df_v_1, on='name', how='inner')
df_2 = pd.merge(df_sy, df_w_2, on='name', how='inner')
df_2 = pd.merge(df_2, df_s_2, on='name', how='inner')
df_2 = pd.merge(df_2, df_v_2, on='name', how='inner')
df_4 = pd.merge(df_sy, df_w_4, on='name', how='inner')
df_4 = pd.merge(df_4, df_s_4, on='name', how='inner')
df_4 = pd.merge(df_4, df_v_4, on='name', how='inner')
df_6 = pd.merge(df_sy, df_w_6, on='name', how='inner')
df_6 = pd.merge(df_6, df_s_6, on='name', how='inner')
df_6 = pd.merge(df_6, df_v_6, on='name', how='inner')


""" CALCULATE """

list_var =   ['SANS_Sum_ENIGMA', 'SANS_Affektverflachung_ENIGMA',
              'SANS_AlogieParalogie_ENIGMA', 'SANS_AbulieApathie_ENIGMA',
              'SANS_Anhedonie_ENIGMA', 'SANS_Aufmerksamkeit_ENIGMA',
              'SANS_Sum_ohneAufmerksamkeit_ENIGMA', 'SAPS_Halluzinationen_ENIGMA',
              'SAPS_Wahn_ENIGMA', 'SAPS_BizarresVerhalten_ENIGMA',
              'SAPS_PositiveFormaleDenkstoerung_ENIGMA', 'SAPS_Sum_ENIGMA',
              'HAMD_Sum17', 'HAMA_Sum', 'YMRS_Sum', 'GAFscore',
              'mean_semsim_t', 'max_semsim_t', 'min_semsim_t',
              'av_ssc_t', 'av_crossing_t', 'autocorr_t',
              'mean_semsim', 'max_semsim', 'min_semsim',
              'av_ssc', 'av_crossing', 'autocorr', 'sum_distances',
              'dispersion_sents',
              'log_volume', 'log_area', 'group']

list_var_1 = ['SANS_Sum_ENIGMA', 'SANS_Affektverflachung_ENIGMA',
              'SANS_AlogieParalogie_ENIGMA', 'SANS_AbulieApathie_ENIGMA',
              'SANS_Anhedonie_ENIGMA', 'SANS_Aufmerksamkeit_ENIGMA',
              'SANS_Sum_ohneAufmerksamkeit_ENIGMA', 'SAPS_Halluzinationen_ENIGMA',
              'SAPS_Wahn_ENIGMA', 'SAPS_BizarresVerhalten_ENIGMA',
              'SAPS_PositiveFormaleDenkstoerung_ENIGMA', 'SAPS_Sum_ENIGMA',
              'HAMD_Sum17', 'HAMA_Sum', 'YMRS_Sum', 'GAFscore']

list_var_2 = ['mean_semsim_t', 'max_semsim_t', 'min_semsim_t',
              'av_ssc_t', 'av_crossing_t', 'autocorr_t',
              'mean_semsim', 'max_semsim', 'min_semsim',
              'av_ssc', 'av_crossing', 'autocorr', 'sum_distances',
              'dispersion_sents',
              'log_volume', 'log_area']

df   = df[list_var]
df_1 = df_1[list_var]
df_2 = df_2[list_var]
df_4 = df_4[list_var]
df_6 = df_6[list_var]

# CORRELATIONS
corr_matrix   = df.corr(numeric_only=True)
corr_matrix_1 = df_1.corr(numeric_only=True)
corr_matrix_2 = df_2.corr(numeric_only=True)
corr_matrix_4 = df_4.corr(numeric_only=True)
corr_matrix_6 = df_6.corr(numeric_only=True)


# PARTIAL CORRELATION
def partial_correlation(df, control_vars=None):
    if control_vars is None:
        control_vars = []
        
    controlled_df = df.copy()
    if control_vars:
        # Convert categorical variables to dummy variables
        control_dummies = pd.get_dummies(df[control_vars], drop_first=True)
        
        for col in controlled_df.columns:
            if col not in control_vars:
                # Combine non-categorical columns with dummies
                non_na = pd.concat([df[col], control_dummies], axis=1).dropna()
                if not non_na.empty:
                    X = non_na[control_dummies.columns]
                    y = non_na[col]
                    reg = LinearRegression().fit(X, y)
                    residuals = y - reg.predict(X)
                    controlled_df[col] = residuals
                else:
                    controlled_df[col] = np.nan
    
    # Compute the correlation matrix
    corr_matrix = controlled_df.corr(numeric_only=True)
    return corr_matrix


# P-VALUES OF PARTIAL CORRELATION
def calculate_p_values_partial(df, control_vars=None):
    df_cols = pd.DataFrame(columns=df.columns)
    pval_matrix = df_cols.transpose().join(df_cols, how='outer')
    if control_vars is None:
        control_vars = []
    for r in df.select_dtypes(include='number').columns:
        for c in df.select_dtypes(include='number').columns:
            if r != c:
                # Drop NaN values
                non_na = df[[r, c] + control_vars].dropna()
                if not non_na.empty and len(non_na) > 1:
                    if control_vars:
                        X = pd.get_dummies(non_na[control_vars], drop_first=True)
                        y_r = non_na[r]
                        y_c = non_na[c]
                        reg_r = LinearRegression().fit(X, y_r)
                        reg_c = LinearRegression().fit(X, y_c)
                        residuals_r = y_r - reg_r.predict(X)
                        residuals_c = y_c - reg_c.predict(X)
                        _, p = pearsonr(residuals_r, residuals_c)
                    else:
                        _, p = pearsonr(non_na[r], non_na[c])
                    pval_matrix.at[r, c] = round(p, 4)
                else:
                    pval_matrix.at[r, c] = float('nan')
            else:
                pval_matrix.at[r, c] = round(0.0, 4)
    return pval_matrix


# P-VALUES MATRIX
def calculate_p_values(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    
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


# CORRELATIONS PLOT
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
    num_var_1 = len(list_var_1)
    num_var_2 = len(list_var_2)
    correlation_matrix = np.zeros((num_var_1, num_var_2))
    for comparison in significant_comparisons:
        var_1_index = list_var_1.index(comparison[0])
        var_2_index = list_var_2.index(comparison[1])
        correlation_matrix[var_1_index, var_2_index] = comparison[2]
     # Map the variable names
    list_var_1_mapped = [variable_names_map.get(var, var) for var in list_var_1]
    list_var_2_mapped = [variable_names_map.get(var, var) for var in list_var_2]
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", 
                xticklabels=list_var_2_mapped, yticklabels=list_var_1_mapped, linewidths=0.5)
    plt.title(f'Correlation coefficients - Picture {bild}', fontsize=16)
    #plt.xlabel('Semsim variables', fontsize=14)
    plt.ylabel('Symptoms', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f'{path_results}heatmap_correlation_semsim_symptoms_{bild}.png', dpi=500, bbox_inches='tight')
    plt.show()
    return significant_comparisons


variable_names_map = {
    'mean_semsim_t': 'mean semsim (BERT)',
    'max_semsim_t': 'max semsim (BERT)',
    'min_semsim_t': 'min semsim (BERT)',
    'av_ssc_t': 'ssc (BERT)',
    'av_crossing_t': 'mean crossing rate (BERT)',
    'autocorr_t': 'autocorrelation (BERT)',
    'mean_semsim': 'mean semsim (fastText)',
    'max_semsim': 'max semsim (fastText)',
    'min_semsim': 'min semsim (fastText)',
    'av_ssc': 'ssc (fastText)',
    'av_crossing': 'mean crossing rate (fastText)',
    'autocorr': 'autocorrelation(fastText)',
    'sum_distances': 'displacement (words)',
    'dispersion_sents': 'dispersion (sents)',
    'log_volume': 'log volume',
    'log_area': 'log area',
    'SANS_Sum_ENIGMA': 'SANS Sum',
    'SANS_Affektverflachung_ENIGMA': 'SANS affective flattening',
    'SANS_AlogieParalogie_ENIGMA': 'SANS alogia/paralogia',
    'SANS_AbulieApathie_ENIGMA': 'SANS abulia/apathy',
    'SANS_Anhedonie_ENIGMA': 'SANS anhedonia',
    'SANS_Aufmerksamkeit_ENIGMA': 'SANS attention',
    'SANS_Sum_ohneAufmerksamkeit_ENIGMA': 'SANS sum without attention',
    'SAPS_Halluzinationen_ENIGMA': 'SAPS hallucinations',
    'SAPS_Wahn_ENIGMA': 'SAPS delusions',
    'SAPS_BizarresVerhalten_ENIGMA': 'SAPS bizarre behavior',
    'SAPS_PositiveFormaleDenkstoerung_ENIGMA': 'SAPS positive formal thought disorder',
    'SAPS_Sum_ENIGMA': 'SAPS Sum',
    'HAMD_Sum17': 'Hamilton depression scale sum score',
    'HAMA_Sum': 'Hamilton anxiety scale sum score',
    'YMRS_Sum': 'Young Mania Rating Scale',
    'GAFscore': 'Global Assessment of Functioning'
    }


# PLOT

# (FIGURE 5)
significant_comparisons   = corr_plot(df, corr_matrix, list_var_1, list_var_2, path_results, 'all')

significant_comparisons_1 = corr_plot(df_1, corr_matrix_1, list_var_1, list_var_2, path_results, '1')
significant_comparisons_2 = corr_plot(df_2, corr_matrix_2, list_var_1, list_var_2, path_results, '2')
significant_comparisons_4 = corr_plot(df_4, corr_matrix_4, list_var_1, list_var_2, path_results, '4')
significant_comparisons_6 = corr_plot(df_6, corr_matrix_6, list_var_1, list_var_2, path_results, '6')


# PARTIAL CORRELATIONS PLOT
def partial_corr_plot(df, corr_matrix, list_var_1, list_var_2, path_results, bild):
    # Calculate P-values
    p_values_matrix = calculate_p_values_partial(df, control_vars)
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
    num_var_1 = len(list_var_1)
    num_var_2 = len(list_var_2)
    correlation_matrix = np.zeros((num_var_1, num_var_2))
    for comparison in significant_comparisons:
        var_1_index = list_var_1.index(comparison[0])
        var_2_index = list_var_2.index(comparison[1])
        correlation_matrix[var_1_index, var_2_index] = comparison[2]
    # Map the variable names
    list_var_1_mapped = [variable_names_map.get(var, var) for var in list_var_1]
    list_var_2_mapped = [variable_names_map.get(var, var) for var in list_var_2]
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", 
                xticklabels=list_var_2_mapped, yticklabels=list_var_1_mapped, linewidths=0.5)
    plt.title('Partial Correlation coefficients', fontsize=16)
    plt.ylabel('Symptoms', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f'{path_results}heatmap_partial_correlation_{bild}.png', dpi=500, bbox_inches='tight')
    plt.show()
    return significant_comparisons


control_vars = ['group']
partial_corr_matrix = partial_correlation(df, control_vars)
partial_p_values_matrix = calculate_p_values_partial(df, control_vars)

# (FIGURE 5)
significant_comparisons   = partial_corr_plot(df, partial_corr_matrix, list_var_1, list_var_2, path_results, 'all')

