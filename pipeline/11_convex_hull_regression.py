# Use the dimension-specific outputs of 06_convex_hull.
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

# MIXED-EFFECTS MODEL
def mixed_model_results(df, y, x):
    df = df.dropna(subset=[y, x, 'group', 'name'])
    model_r = MixedLM.from_formula(f'{y} ~ group + {x} + bild', df, groups=df["name"])
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


for dimension in range(3, 11):
    print(f"Convex hull regression: {dimension}D")
    df_sents = pd.read_pickle(path_results+f'df_sent_convexhull_{dimension}D_mono.pkl')
    df_sents['log_n_sent'] = np.log(df_sents['n_sent'])
    df_sents['log_volume'] = np.log(df_sents['volume'])
    df_sents['log_area']   = np.log(df_sents['area'])

    mixed_model_results(df_sents, 'log_volume', 'n_sent')
    mixed_model_results(df_sents, 'log_area', 'n_sent')
