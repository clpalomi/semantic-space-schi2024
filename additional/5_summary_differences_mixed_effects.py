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
import matplotlib.pyplot as plt
from statsmodels.regression.mixed_linear_model import MixedLM


# PATH INPUT
path_input = ADDITIONAL_RESULTS
path_results = SUMMARIES


""" FUNCTIONS """

def get_summary_mixed(file, var, task, name):
    df =  pd.read_pickle(file)
    model = MixedLM.from_formula(f'{var} ~ group + length + {task}', df, groups=df["name"])
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
    sns.pointplot(x=f'{task}', y=f'{var}', hue='group', data=df, errorbar=None, 
                  markers=['s', 'o', 'v'], palette=color_palette, 
                  ax=ax2)
    ax2.set_xlabel('Task', fontsize=15)
    ax2.set_ylabel(f'{var}', fontsize=15)
    ax2.legend(fontsize=15)
    ax2.set_title(f'Interaction Plot of Group and Task on {var}', fontsize=17)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    #plt.savefig(path_results+f'{name}_{var}.jpg', dpi=1200, bbox_inches='tight')
    #plt.savefig(path_results+f'{name}_{var}.eps', dpi=1200, bbox_inches='tight')
    plt.show()
    return result.summary()


file = path_input+'df_semsim_all.pkl'
df =  pd.read_pickle(file)

get_summary_mixed(file, 'mean', 'bild', 'semsim')


""" MAIN """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("base", type=str)
    parser.add_argument("var", type=str)
    parser.add_argument("task", type=str)
    parser.add_argument("name", type=str)
    
    args  = parser.parse_args()
    base  = args.base
    var   = args.var
    task  = args.task
    name  = args.name
    
    result = get_summary_mixed(path_input+base, var, task, name)
    print(result)
    del result
    
    
    
    
    
