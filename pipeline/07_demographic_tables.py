from pipeline.paths import METADATA, RESULTS, TABLES

"""
@author: Claudio

Tables and results used in the article

"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

# PATH
path_demo = METADATA
path_tables = TABLES
path_results = RESULTS

""" 1. DEMOGRAPHICS (TABLE 1.1) """ 

# ALL DEMOGRAPHICS AND COREFERENCE SCORES
df = pd.read_csv(path_demo+'ssd_mdd_hc_data.csv')
# columns for demographics
columns = ['Proband','Typ2', 'Alter', 'Geschlecht', 'Bildungsjahre', 'IQ', 'AgeOfOnset']
df = df[columns]
# change names
column_names = {'Proband': 'participant', 'Typ2': 'group', 'Alter': 'age',
                'Geschlecht': 'gender', 'Bildungsjahre': 'years_ed'}
df = df.rename(columns=column_names)
# gender
df['gender'] = df['gender'].replace([1, 2], ['M', 'F'])
# replace -99 values with NaN
df.replace(-99, np.nan, inplace=True)
grouped = df.groupby('group').agg({'group': 'count',
                                   'age': [lambda x: x.mean(skipna=True), lambda x: x.std(skipna=True)],
                                   'gender': lambda x: x.value_counts().get('F', 0) / len(x),
                                   'years_ed': [lambda x: x.mean(skipna=True), lambda x: x.std(skipna=True)],
                                   'IQ':  [lambda x: x.mean(skipna=True), lambda x: x.std(skipna=True)],
                                   'AgeOfOnset': [lambda x: x.mean(skipna=True), lambda x: x.std(skipna=True)]})

grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
grouped.columns = [
    'number', 
    'age_mean', 'age_std', 
    'proportion_female', 
    'years_ed_mean', 'years_ed_std', 
    'IQ_mean', 'IQ_std', 
    'AgeOfOnset_mean', 'AgeOfOnset_std'
]
grouped = grouped.T
# TO EXCEL
grouped.to_excel(path_tables+'table_1.xlsx', index=True)


""" 2. DEMOGRAPHICS (TABLE 1.2) """ 

# ttest
def perform_ttest(data1, data2):
    data1 = data1.dropna()
    data2 = data2.dropna()
    t_statistic, p_value = ttest_ind(data1, data2)
    return t_statistic, p_value

# chi-square
def perform_chi_square(data):
    chi2_stat, p_value, _, _ = chi2_contingency(pd.crosstab(data['group'], data['gender']))
    return chi2_stat, p_value

# compare results
def compare_groups(df, variable):
    results = []
    groups = df['group'].unique()
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            group1 = groups[i]
            group2 = groups[j]
            data1 = df[df['group'] == group1][variable]
            data2 = df[df['group'] == group2][variable]
            if variable == 'gender':
                statistic, p_value = perform_chi_square(df[df['group'].isin([group1, group2])])
                test_name = 'Chi-square'
            else:
                statistic, p_value = perform_ttest(data1, data2)
                test_name = 'T-test'
            comparison = f'{group1} - {group2}'
            results.append([variable, comparison, test_name, statistic, p_value])
    return results

# Perform comparisons
results = []
variables = ['age', 'gender', 'years_ed', 'IQ']
for var in variables:
    results.extend(compare_groups(df, var))

compare_df = pd.DataFrame(results, columns=['Variable','Comparison', 'Test', 'Statistic', 'p_value'])
# TO EXCEL
compare_df.to_excel(path_tables+'table_2.xlsx', index=False)

