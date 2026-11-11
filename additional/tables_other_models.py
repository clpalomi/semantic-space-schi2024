# Interactive model variants with unresolved variables; see additional/README.md.
from pipeline.paths import METADATA, TABLES, ADDITIONAL_RESULTS

"""
@author: Claudio

Tables and results used in the article

"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
from scipy.stats import pearsonr

# PATH
path_demo = METADATA
path_tables = TABLES
path_results = ADDITIONAL_RESULTS

""" 9. TABLE XX """
from statsmodels.regression.mixed_linear_model import MixedLM

# READ VARIABLES WORD EMBEDDINGS
df_words = pd.read_pickle(path_results+'df_semsim_dist_length.pkl')
df_sents = pd.read_pickle(path_results+'df_semsim_sent_length.pkl')

# MIXED-EFFECTS MODEL
def mixed_model_results(df, y, x):
    model_r = MixedLM.from_formula(f'{y} ~ group + {x} + sent_av_length + bild', df, groups=df["name"])
    result  = model_r.fit()
    # Calculate R-squared and adjusted R-squared
    residual_variance = np.var(result.resid)
    # Calculate total variance of dependent variable
    total_variance = np.var(df[y])
    # Calculate R-squared
    r_squared = 1 - (residual_variance / total_variance)
    # Number of observations
    n = len(df)
    # Number of predictors
    p = len(result.params)-2
    # Calculate adjusted R-squared
    r_squared_adj = 1 - ((1 - r_squared) * ((n - 1) / (n - p - 1)))
    print(result.summary())
    print(f"R-squared: {r_squared}")
    print(f"Adjusted R-squared: {r_squared_adj}")
    return [result, r_squared, r_squared_adj]

# CALCULATE 1
variable = 'sum_distances'
other    = 'n_content'
result_1, r_squared_1, r_squared_adj_1 = mixed_model_results(df_words, variable, other)

# CALCULATE 2
variable = 'mean_semsim'
other    = 'n_content'
result_2, r_squared_2, r_squared_adj_2 = mixed_model_results(df_words, variable, other)

# CALCULATE 2.2
variable = 'dispersion'
other    = 'n_content'
result_2, r_squared_2, r_squared_adj_2 = mixed_model_results(df_words, variable, other)

# CALCULATE 3
variable = 'dispersion'
other    = 'n_sent'
result_3, r_squared_3, r_squared_adj_3 = mixed_model_results(df_sents, variable, other)

# CALCULATE 4
variable = 'sum_dist'
other    = 'n_sent'
result_3, r_squared_3, r_squared_adj_3 = mixed_model_results(df_sents, variable, other)



"""
results = []
coefficients = result.params
std_errors = result.bse
z_values = result.tvalues
p_values = result.pvalues
conf_int = result.conf_int()
# Append results to list
results.append({'Dependent Variable': variable,
                'Independent Variables': '+'.join(['group'] + [f'{other}'] + ['sent_av_length'] + ['bild']),
                'Coefficients': coefficients,
                'Standard Error': std_errors,
                'z-value': z_values,
                'p-value': p_values,
                '95% Confidence Interval': conf_int,
                'R-squared': r_squared,
                'Adjusted R-squared': r_squared_adj
                })
df = pd.DataFrame(results)
"""















"""  OTHERS """

# IMPORT ALL DATA
df_demo = pd.read_csv(path_demo+'ssd_mdd_hc_data.csv')

# KEEP RESULTS MODELS AND GROUPS
df_demo = df_demo[['Proband', 'SANS_Sum_ENIGMA', 
                   'SAPS_PositiveFormaleDenkstoerung_ENIGMA',
                   'SAPS_Sum_ENIGMA', 'IQ']]
# REPLACE -99 AS NAN
df_demo.replace([-99, -93], np.nan, inplace=True)

# RENAME
df_demo.rename(columns={'Proband': 'name', 'SANS_Sum_ENIGMA': 'SANS_Sum',
                        'SAPS_PositiveFormaleDenkstoerung_ENIGMA': 'SAPS_FTD',
                        'SAPS_Sum_ENIGMA': 'SAPS_Sum'}, inplace=True)

# MERGE
df_sents = df_sents.merge(df_demo, on='name', how='left')
# df = df.merge(df_demo, on='name', how='left')





""" CORRECTED P-VALUES """
from statsmodels.regression.mixed_linear_model import MixedLM

# READ VARIABLES WORD EMBEDDINGS
df_words   = pd.read_pickle(path_results+'df_semsim_dist_length.pkl')
df_sents_1 = pd.read_pickle(path_results+'df_semsim_sent_length.pkl')
df_sents_2 = pd.read_pickle(path_results+'df_sent_convexhull_5D.pkl')


# MIXED-EFFECTS MODEL
def mixed_model_results_words(df, y, x):
    model_r = MixedLM.from_formula(f'{y} ~ group + {x} + sent_av_length + bild', df, groups=df["name"])
    result  = model_r.fit()
    # Calculate R-squared and adjusted R-squared
    residual_variance = np.var(result.resid, ddof=1) 
    total_variance = np.var(df[y], ddof=1)
    r_squared = 1 - (residual_variance / total_variance)
    # Number of observations
    n = len(df)
    # Number of predictors
    p = len(result.params)-2
    # Calculate adjusted R-squared
    r_squared_adj = 1 - ((1 - r_squared) * ((n - 1) / (n - p - 1)))
    print(result.summary())
    print(f"R-squared: {r_squared}")
    print(f"Adjusted R-squared: {r_squared_adj}")
    # Compute standardized coefficients
    std_coeffs = {}
    for predictor in result.params.index:
        if predictor == 'Intercept':
            continue  # Skip the intercept
        # Standard deviation for dummy variable is calculated as for any variable
        std_X = df[predictor].std() if predictor in df.columns else 1
        std_Y = df[y].std()
        beta = result.params[predictor]
        std_beta = beta * (std_X / std_Y)
        std_coeffs[predictor] = std_beta
    print("Standardized Coefficients:", std_coeffs)
    return [result, r_squared, r_squared_adj, std_coeffs]


# MIXED-EFFECTS MODEL EUCLIDEAN
def mixed_model_results_euclidean(df, y, x):
    model_r = MixedLM.from_formula(f'{y} ~ group + {x} + sent_av_length + bild', df, groups=df["name"])
    result  = model_r.fit()
    # Calculate R-squared and adjusted R-squared
    residual_variance = np.var(result.resid)
    # Calculate total variance of dependent variable
    total_variance = np.var(df[y])
    # Calculate R-squared
    r_squared = 1 - (residual_variance / total_variance)
    # Number of observations
    n = len(df)
    # Number of predictors
    p = len(result.params)-2
    # Calculate adjusted R-squared
    r_squared_adj = 1 - ((1 - r_squared) * ((n - 1) / (n - p - 1)))
    print(result.summary())
    print(f"R-squared: {r_squared}")
    print(f"Adjusted R-squared: {r_squared_adj}")
    return [result, r_squared, r_squared_adj]


# MIXED-EFFECTS MODEL
def mixed_model_results_convex(df, y):
    df['n_sent_quantile'] = pd.qcut(df['n_sent'], q=[0, 0.5, 1], labels=False)
    df = add_constant(df)
    formula = f'{y} ~ C(group) + C(n_sent_quantile)'
    # formula = f'{y} ~ C(group) + C(bild)'
    model_r = MixedLM.from_formula(formula, df, groups=df["name"])
    
    #model_r = MixedLM.from_formula(f'{y} ~ C(group) + C(bild)', df, groups=df["name"])
    #model_r = MixedLM.from_formula(f'{y} ~ group + {x}', df, groups=df["name"])
    result  = model_r.fit()
    # Calculate R-squared and adjusted R-squared
    residual_variance = np.var(result.resid)
    # Calculate total variance of dependent variable
    total_variance = np.var(df[y])
    # Calculate R-squared
    r_squared = 1 - (residual_variance / total_variance)
    # Number of observations
    n = len(df)
    # Number of predictors
    p = len(result.params)-2
    # Calculate adjusted R-squared
    r_squared_adj = 1 - ((1 - r_squared) * ((n - 1) / (n - p - 1)))
    print(result.summary())
    print(f"R-squared: {r_squared}")
    print(f"Adjusted R-squared: {r_squared_adj}")
    return [result, r_squared, r_squared_adj]




# LOOP FOR ALL THE VARIABLES
variables = ['mean_semsim', 'max_semsim', 'min_semsim',
             'av_ssc', 'av_crossing', 'autocorr',
             'mean_semsim_t', 'max_semsim_t', 'min_semsim_t',
             'av_ssc_t', 'av_crossing_t', 'autocorr_t']

# FastText variables
variables_f = ['mean_semsim', 'max_semsim', 'min_semsim',
             'av_ssc', 'av_crossing', 'autocorr']

# BERT variables
variables_b = ['mean_semsim_t', 'max_semsim_t', 'min_semsim_t',
             'av_ssc_t', 'av_crossing_t', 'autocorr_t']


# loop

variable = 'mean_semsim'
other    = 'n_content'

result, r_squared, r_squared_adj, std_coeffs = mixed_model_results_words(df_words, variable, other)





# CALCULATE 1
variable = 'sum_distances'
other    = 'n_content'
result_1, r_squared_1, r_squared_adj_1 = mixed_model_results_euclidean(df_words, variable, other)

# CALCULATE 2
variable = 'mean_semsim'
other    = 'n_content'
result_2, r_squared_2, r_squared_adj_2 = mixed_model_results_euclidean(df_words, variable, other)

# CALCULATE 2.2
variable = 'dispersion'
other    = 'n_content'
result_2, r_squared_2, r_squared_adj_2 = mixed_model_results_euclidean(df_words, variable, other)

# CALCULATE 3
variable = 'dispersion'
other    = 'n_sent'
result_3, r_squared_3, r_squared_adj_3 = mixed_model_results_euclidean(df_sents, variable, other)

# CALCULATE 4
variable = 'sum_dist'
other    = 'n_sent'
result_3, r_squared_3, r_squared_adj_3 = mixed_model_results_euclidean(df_sents, variable, other)



[res_v, rs_v, rsa_v] = mixed_model_results_convex(df_sents_2, 'volume_n')
[res_a, rs_a, rsa_a] = mixed_model_results_convex(df_sents_1, 'area_n')

