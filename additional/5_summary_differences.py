# Additional analysis; check additional/README.md before running.
from pipeline.paths import ADDITIONAL_RESULTS, SUMMARIES

"""
Created on Wed Sep 20 23:06:11 2023
@author: Claudio
"""

# IMPORT PACKAGES
import argparse
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from scipy import stats

# PATH INPUT
path_input = ADDITIONAL_RESULTS
path_results = SUMMARIES



""" FUNCTIONS """

# GET A COMPARISON SUMMARY
def get_summary(file, groups, name):
    df = pd.read_pickle(file)
    num_columns = df.select_dtypes(include=['number'])
    s_group_column = df[['s_group']]
    # Only variables and sub group column
    df = pd.concat([num_columns, s_group_column], axis=1)
    # preserve if belong to groups of comparison
    df = df[df['s_group'].isin(groups)]
    summary_df = pd.DataFrame(columns=
                              ['Variable', groups[0], groups[1], 
                               'Difference', 'P-value', 'Effect Size'])
    for col in df.columns:
        if col != 's_group':  # Exclude the 'group' column
            # Perform ANOVA to compare groups
            formula = f"{col} ~ C(s_group)"
            model = ols(formula, data=df).fit()
            aov_table = sm.stats.anova_lm(model, typ=2)
            # Calculate averages for each group
            averages = df.groupby('s_group')[col].mean()
            # Calculate difference
            diff = averages.diff().iloc[-1]
            # Extract p-value
            p_value = aov_table.loc['C(s_group)', 'PR(>F)']
            # Calculate effect size (you can use other measures like Cohen's d)
            effect_size = (averages[1] - averages[0]) / df[col].std()
            # Concat the results with the summary DataFrame
            df_comp = pd.DataFrame([{
                'Variable': col,
                groups[0]: averages[groups[0]],
                groups[1]: averages[groups[1]],
                'Difference': diff,
                'P-value': p_value,
                'Effect Size': effect_size
                }])
            summary_df = pd.concat([summary_df, df_comp], ignore_index=True)
    # Export to Excel
    summary_df.to_excel(path_results + name +'.xlsx', index=False)
    return summary_df


#  BOX-PLOT WIHT TTEST ALL VARIABLES
def plot_with_ttest(file, groups, name):
    df = pd.read_pickle(file)
    num_columns = df.select_dtypes(include=['number'])
    s_group_column = df[['s_group']]
    # Only variables and sub group column
    df = pd.concat([num_columns, s_group_column], axis=1)
    # preserve if belong to groups of comparison
    df = df[df['s_group'].isin(groups)]
    # s_group as x
    x = 's_group'
    # VARIABLES
    variables = list(num_columns.columns)
    # LOOP OVER VARIABLES TO GRAPH 
    for y in variables:
        plot_with_ttest_var(x, y, df, groups, name)
        

#  BOX-PLOT WIHT TTEST - VARIABLE Y
def plot_with_ttest_var(x, y, df, groups, name):
    # GRAPHS AND DIFFERENCES
    sns.boxplot(x=x, y=y, data=df, showfliers=False)
    # Perform t-test
    group1 = df[df[x] == groups[0]][y]
    group2 = df[df[x] == groups[1]][y]
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
    plt.savefig(path_results + name+'_'+y+'.eps', dpi=1200, bbox_inches='tight')
    plt.savefig(path_results + name+'_'+y+'.jpg', dpi=1200, bbox_inches='tight')
    plt.close()



""" MAIN """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("base", type=str)
    parser.add_argument("grupo1", type=str)
    parser.add_argument("grupo2", type=str)
    parser.add_argument("name", type=str)
    
    args    = parser.parse_args()
    base    = args.base
    grupo1  = args.grupo1
    grupo2  = args.grupo2
    name    = args.name
    
    result = get_summary(path_input+base, [grupo1, grupo2], name)
    print(result)
    plot_with_ttest(path_input+base, [grupo1, grupo2], name)
    del result
