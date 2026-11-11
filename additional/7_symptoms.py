# Additional analysis; check additional/README.md before running.
from pipeline.paths import METADATA, ADDITIONAL_RESULTS

# -*- coding: utf-8 -*-
"""
@author: Claudio
"""

# IMPORT PACKAGES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

# PATH INPUT
path_input = METADATA
path_results = ADDITIONAL_RESULTS

df = pd.read_pickle(ADDITIONAL_RESULTS+'df_semsim_all.pkl')

# MERGE WITH SYMPTOMS
df_s = pd.read_excel(path_input+'symptoms.xlsx')
df_s = df_s.replace(-99, np.nan)
df_s = df_s.replace(-93, np.nan)
df = pd.merge(df, df_s, on='name', how='inner')

# ONLY SZ
df = df[df['group']=='Sz']

# Display column names
column_names = df.columns
print(column_names)

# VAR
list_var = ['SANS_Sum_ENIGMA', 'SANS_Affektverflachung_ENIGMA',
'SANS_AlogieParalogie_ENIGMA', 'SANS_AbulieApathie_ENIGMA',
'SANS_Anhedonie_ENIGMA', 'SANS_Aufmerksamkeit_ENIGMA',
'SANS_Sum_ohneAufmerksamkeit_ENIGMA', 'SAPS_Halluzinationen_ENIGMA',
'SAPS_Wahn_ENIGMA', 'SAPS_BizarresVerhalten_ENIGMA',
'SAPS_PositiveFormaleDenkstoerung_ENIGMA', 'SAPS_Sum_ENIGMA']


# NUMBER OF NAN AND ZEROS
nan_counts  = {}
zero_counts = {}
for var in list_var:
    nan_count  = df[var].isnull().sum()
    zero_count = (df[var] == 0).sum()
    nan_counts[var]  = nan_count
    zero_counts[var] = zero_count
nan_table = pd.DataFrame(list(nan_counts.items()), columns=['Variable', 'NaN Count'])
nan_table['NaN Count'] = (nan_table['NaN Count'] / 4).round(0)
nan_table['Zero Count'] = nan_table['Variable'].map(zero_counts)
nan_table['Zero Count'] = (nan_table['Zero Count'] / 4).round(0)
nan_table.to_excel(path_results+'nan_symptoms_table.xlsx', index=False)

# SCATTER PLOTS
def scatter_plots(df, list_var, bild_values):
    for bild_value in bild_values:
        filtered_df = df[df['bild'] == bild_value]
        for var in list_var:
            correlation = filtered_df[['mean', var]].corr().iloc[0, 1]
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=filtered_df, x='mean', y=var, marker='o', label=f'Corr: {correlation:.2f}')
            plt.title(f'Scatter Plot for {var} and Mean (task={bild_value})', fontsize=16)
            plt.xlabel('mean', fontsize=14)
            plt.ylabel(var, fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.savefig(path_results+f'scatter_plot_{var}_mean_task_{bild_value}.png')
            plt.show()

bild_values = ['1', '2', '4', '6']
scatter_plots(df, list_var, bild_values)



""" MIXED MODEL """

def run_mixed_model(variable, df):
    formula = f'{variable} ~ mean * bild'
    model = MixedLM.from_formula(formula, df, groups=df["name"])
    result = model.fit()
    print(f"\nResults for {variable}:\n")
    print(result.summary())
    

# DROP IF TOO MANY NAN VALUES
n_obs1 = len(df)   # 165 observaciones
nan_counts = df.isna().sum(axis=1)
df = df[nan_counts < 8]
n_obs2 = len(df)   # 148
df.dropna(inplace=True)  # 96
df = df.reset_index()


for var in list_var:
    run_mixed_model(var, df)
