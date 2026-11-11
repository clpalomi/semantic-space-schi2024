from pipeline.paths import METADATA, RESULTS, TABLES

"""
@author: Claudio

Tables and results used in the article

"""

import pandas as pd
import numpy as np

# PATH
path_demo = METADATA
path_tables = TABLES
path_results = RESULTS

from statsmodels.regression.mixed_linear_model import MixedLM

# READ VARIABLES WORD EMBEDDINGS
df_words = pd.read_pickle(path_results+'df_semsim_dist_length.pkl')
df_sents = pd.read_pickle(path_results+'df_semsim_sent_length.pkl')

# MIXED-EFFECTS MODEL
def mixed_model_results(df, y, x):
    model_r = MixedLM.from_formula(f'{y} ~ group + {x} + bild + sent_av_length', df, groups=df["name"])
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
    p = len(result.fe_params)
    # Calculate adjusted R-squared
    r_squared_adj = 1 - ((1 - r_squared) * ((n - 1) / (n - p - 1)))
    # df_residual = number of observations - number of fixed effects parameters
    df_residual = n - p
    print(result.summary())
    print(f"R-squared: {r_squared}")
    print(f"Adjusted R-squared: {r_squared_adj}")
    print(f"Degrees of Freedom (Residual): {df_residual}")
    return [result, r_squared, r_squared_adj, df_residual]

# (TABLE 1)
variable = 'sum_distances'
other    = 'n_content'
mixed_model_results(df_words, variable, other)

# (TABLE 2)
variable = 'dispersion'
other    = 'n_sent'
mixed_model_results(df_sents, variable, other)

# (TABLE S5)
variable = 'sum_distances_n'
other    = 'n_content'
mixed_model_results(df_words, variable, other)

# (TABLE S6)
variable = 'sum_dist'
other    = 'n_sent'
mixed_model_results(df_sents, variable, other)

