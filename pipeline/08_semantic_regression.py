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

# Read dataframe of values
df_words = pd.read_pickle(path_results+'df_semsim_dist_length.pkl')

# MIXED-EFFECTS MODEL
def mixed_model_results(df, y, x):
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

results = []
for variable in variables:
    if variable in variables_f:
        other = 'n_content'
    if variable in variables_b:
        other = 'total_length'
    result, r_squared, r_squared_adj, std_coeffs = mixed_model_results(df_words, variable, other)
    coefficients = result.params
    std_errors = result.bse
    z_values = result.tvalues
    p_values = result.pvalues
    conf_int = result.conf_int()
    # Append results to list
    results.append({
        'Dependent Variable': variable,
        'Independent Variables': '+'.join(['group'] + [f'{other}'] + ['sent_av_length'] + ['bild']),
        'Coefficients': coefficients,
        'Standard Error': std_errors,
        'z-value': z_values,
        'p-value': p_values,
        '95% Confidence Interval': conf_int,
        'R-squared': r_squared,
        'Adjusted R-squared': r_squared_adj,
        'std_coeffs': std_coeffs        
    })
df = pd.concat([pd.DataFrame([r]) for r in results], ignore_index=True)

# RESULTS IN A TABLE
new_df = pd.DataFrame(columns=['std_coeffs', 'p-value'])

# ONLY STD COEFFS AND P-VALUES
for i in range(12):
    if i < 6:
        other = 'n_content'
    else:
        other = 'total_length'
    row_data = {
        'std_coeffs': [
            df['std_coeffs'][i]['group[T.MDD]'],
            df['std_coeffs'][i]['group[T.SSD]'],
        ],
        'p-value': [
            df['p-value'][i]['group[T.MDD]'],
            df['p-value'][i]['group[T.SSD]'],
        ]
    }
    # Concatenate the row to the new DataFrame
    new_df = pd.concat([new_df, pd.DataFrame(row_data)], ignore_index=True)

# Add independent variable
independent_variables  = ['MDD', 'SSD']
independent_variables_v = independent_variables * (len(new_df) // len(independent_variables)) + independent_variables[:len(new_df) % len(independent_variables)]
new_df['Independent variable'] = independent_variables_v

# Add dependent variable
dependent_variables = ['mean semsim', 'max semsim', 'min semsim', 'average ssc', 'average crossing', 'autocorrelation',
                       'mean semsim t', 'max semsim t', 'min semsim t', 'average ssc t', 'average crossing t', 'autocorrelation t']
dependent_variable_values = [var for var in dependent_variables for _ in range(2)]
new_df['Dependent variable'] = dependent_variable_values[:len(new_df)]

# re-order
new_df = new_df[['Dependent variable', 'Independent variable', 'std_coeffs', 'p-value']]

# TO EXCEL
new_df.to_excel(path_results+'regression_table_signif.xlsx', index=False)


# ALL VARIABLES
new_df = pd.DataFrame(columns=['Dependent variable', 'Independent variable', 
                               'Coefficients', 'Standard Error', 'z-value', 'p-value', 
                               'Confidence Interval 0.025', 'Confidence Interval 0.975',
                               'R-squared', 'Adjusted R-squared'])
for i in range(12):
    if i < 6:
        other = 'n_content'
    else:
        other = 'total_length'
    row_data = {
        'Coefficients': [
            df['Coefficients'][i]['Intercept'],
            df['Coefficients'][i]['group[T.MDD]'],
            df['Coefficients'][i]['group[T.SSD]'],
            df['Coefficients'][i]['bild[T.2]'],
            df['Coefficients'][i]['bild[T.4]'],
            df['Coefficients'][i]['bild[T.6]'],
            df['Coefficients'][i][f'{other}'],
            df['Coefficients'][i]['sent_av_length'],
            df['Coefficients'][i]['Group Var']
        ],
        'Standard Error': [
            df['Standard Error'][i]['Intercept'],
            df['Standard Error'][i]['group[T.MDD]'],
            df['Standard Error'][i]['group[T.SSD]'],
            df['Standard Error'][i]['bild[T.2]'],
            df['Standard Error'][i]['bild[T.4]'],
            df['Standard Error'][i]['bild[T.6]'],
            df['Standard Error'][i][f'{other}'],
            df['Standard Error'][i]['sent_av_length'],
            df['Standard Error'][i]['Group Var']
        ],
        'z-value': [
            df['z-value'][i]['Intercept'],
            df['z-value'][i]['group[T.MDD]'],
            df['z-value'][i]['group[T.SSD]'],
            df['z-value'][i]['bild[T.2]'],
            df['z-value'][i]['bild[T.4]'],
            df['z-value'][i]['bild[T.6]'],
            df['z-value'][i][f'{other}'],
            df['z-value'][i]['sent_av_length'],
            df['z-value'][i]['Group Var']
        ],
        'p-value': [
            df['p-value'][i]['Intercept'],
            df['p-value'][i]['group[T.MDD]'],
            df['p-value'][i]['group[T.SSD]'],
            df['p-value'][i]['bild[T.2]'],
            df['p-value'][i]['bild[T.4]'],
            df['p-value'][i]['bild[T.6]'],
            df['p-value'][i][f'{other}'],
            df['p-value'][i]['sent_av_length'],
            df['p-value'][i]['Group Var']
        ],
        'Confidence Interval 0.025': [
            df['95% Confidence Interval'][i][0]['Intercept'],
            df['95% Confidence Interval'][i][0]['group[T.MDD]'],
            df['95% Confidence Interval'][i][0]['group[T.SSD]'],
            df['95% Confidence Interval'][i][0]['bild[T.2]'],
            df['95% Confidence Interval'][i][0]['bild[T.4]'],
            df['95% Confidence Interval'][i][0]['bild[T.6]'],
            df['95% Confidence Interval'][i][0][f'{other}'],
            df['95% Confidence Interval'][i][0]['sent_av_length'],
            df['95% Confidence Interval'][i][0]['Group Var']
        ],
        'Confidence Interval 0.975': [
            df['95% Confidence Interval'][i][1]['Intercept'],
            df['95% Confidence Interval'][i][1]['group[T.MDD]'],
            df['95% Confidence Interval'][i][1]['group[T.SSD]'],
            df['95% Confidence Interval'][i][1]['bild[T.2]'],
            df['95% Confidence Interval'][i][1]['bild[T.4]'],
            df['95% Confidence Interval'][i][1]['bild[T.6]'],
            df['95% Confidence Interval'][i][1][f'{other}'],
            df['95% Confidence Interval'][i][1]['sent_av_length'],
            df['95% Confidence Interval'][i][1]['Group Var']
        ],
        'R-squared': [
            df['R-squared'][i],
            df['R-squared'][i],
            df['R-squared'][i],
            df['R-squared'][i],
            df['R-squared'][i],
            df['R-squared'][i],
            df['R-squared'][i],
            df['R-squared'][i],
            df['R-squared'][i]
        ],
        'Adjusted R-squared': [
            df['Adjusted R-squared'][i],
            df['Adjusted R-squared'][i],
            df['Adjusted R-squared'][i],
            df['Adjusted R-squared'][i],
            df['Adjusted R-squared'][i],
            df['Adjusted R-squared'][i],
            df['Adjusted R-squared'][i],
            df['Adjusted R-squared'][i],
            df['Adjusted R-squared'][i]
        ]
    }
    # Concatenate the row to the new DataFrame
    new_df = pd.concat([new_df, pd.DataFrame(row_data)], ignore_index=True)

# Add independent variable
independent_variables  = ['Intercept', 'MDD', 'SSD', 
                       'Picture 2', 'Picture 4', 'Picture 6',
                       'Word count', 'Average sentence length', 'Group Var']
independent_variables_v = independent_variables * (len(new_df) // len(independent_variables)) + independent_variables[:len(new_df) % len(independent_variables)]
new_df['Independent variable'] = independent_variables_v

# Add dependent variable
dependent_variables = ['mean semsim', 'max semsim', 'min semsim', 'average ssc', 'average crossing', 'autocorrelation',
                       'mean semsim t', 'max semsim t', 'min semsim t', 'average ssc t', 'average crossing t', 'autocorrelation t']
dependent_variable_values = [var for var in dependent_variables for _ in range(9)]
new_df['Dependent variable'] = dependent_variable_values[:len(new_df)]

# Confidencde interval
new_df['95% Confidence Interval'] = '(' + new_df['Confidence Interval 0.025'].map('{:.4f}'.format) + ', ' + new_df['Confidence Interval 0.975'].map('{:.4f}'.format) + ')'

# re-order
new_df = new_df[['Dependent variable', 'Independent variable', 'Coefficients',
       'Standard Error', 'z-value', 'p-value', '95% Confidence Interval',
       'R-squared', 'Adjusted R-squared']]

# TO EXCEL
new_df.to_excel(path_results+'regression_table.xlsx', index=False)

