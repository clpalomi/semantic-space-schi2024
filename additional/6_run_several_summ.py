# Additional analysis; check additional/README.md before running.
# -*- coding: utf-8 -*-
"""
@author: Claudio

CALL TO SUMMARY DIFFERENCES FOR DIFFERENT ARGUMENTS
"""


# > python 3_summary_differences.py "base" "group1" "group2" "name_file"

import subprocess
import itertools


# RETURN POSSIBLE COMBINATIONS OF GROUPS
def gen_arguments(arg1, name, groups):
    combinations = list(itertools.combinations(groups, 2))
    arguments_list = [(arg1, group1, group2, 
                       f"{name}_{group1}_{group2}") for group1, group2 in combinations]
    return arguments_list



""" TUPLES OF ARGUMENTS """

# NETWORK
arg1 = "umap/df_n_all.pkl"
name = "summ_n_all"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_1 = gen_arguments(arg1, name, groups)

arg1 = "umap/df_n_av.pkl"
name = "summ_n_av"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_2 = gen_arguments(arg1, name, groups)

arg1 = "umap/df_n_1.pkl"
name = "summ_n_1"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_3 = gen_arguments(arg1, name, groups)

arg1 = "umap/df_n_2.pkl"
name = "summ_n_2"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_4 = gen_arguments(arg1, name, groups)

arg1 = "umap/df_n_4.pkl"
name = "summ_n_4"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_5 = gen_arguments(arg1, name, groups)

arg1 = "umap/df_n_6.pkl"
name = "summ_n_6"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_6 = gen_arguments(arg1, name, groups)


# SEMSIM
arg1 = "df_semsim_all.pkl"
name = "summ_semsim_all"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_7 = gen_arguments(arg1, name, groups)

arg1 = "df_semsim_av.pkl"
name = "summ_semsim_av"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_8 = gen_arguments(arg1, name, groups)

arg1 = "df_semsim_1.pkl"
name = "summ_semsim_1"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_9 = gen_arguments(arg1, name, groups)

arg1 = "df_semsim_2.pkl"
name = "summ_semsim_2"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_10 = gen_arguments(arg1, name, groups)

arg1 = "df_semsim_4.pkl"
name = "summ_semsim_4"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_11 = gen_arguments(arg1, name, groups)

arg1 = "df_semsim_6.pkl"
name = "summ_semsim_6"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_12 = gen_arguments(arg1, name, groups)


# PERPLEXITY
arg1 = "df_perplexity_all.pkl"
name = "summ_perplexity_all"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_13 = gen_arguments(arg1, name, groups)

arg1 = "df_perplexity_av.pkl"
name = "summ_perplexity_av"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_14 = gen_arguments(arg1, name, groups)

arg1 = "df_perplexity_1.pkl"
name = "summ_perplexity_1"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_15 = gen_arguments(arg1, name, groups)

arg1 = "df_perplexity_2.pkl"
name = "summ_perplexity_2"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_16 = gen_arguments(arg1, name, groups)

arg1 = "df_perplexity_4.pkl"
name = "summ_perplexity_4"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_17 = gen_arguments(arg1, name, groups)

arg1 = "df_perplexity_6.pkl"
name = "summ_perplexity_6"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_18 = gen_arguments(arg1, name, groups)


# PERPLEXITY TYPE
arg1 = "df_perplexity_type_all.pkl"
name = "summ_perplexity_type_all"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_19 = gen_arguments(arg1, name, groups)

arg1 = "df_perplexity_type_av.pkl"
name = "summ_perplexity_type_av"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_20 = gen_arguments(arg1, name, groups)

arg1 = "df_perplexity_type_1.pkl"
name = "summ_perplexity_type_1"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_21 = gen_arguments(arg1, name, groups)

arg1 = "df_perplexity_type_2.pkl"
name = "summ_perplexity_type_2"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_22 = gen_arguments(arg1, name, groups)

arg1 = "df_perplexity_type_4.pkl"
name = "summ_perplexity_type_4"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_23 = gen_arguments(arg1, name, groups)

arg1 = "df_perplexity_type_6.pkl"
name = "summ_perplexity_type_6"
groups = ['HC', 'Sz', 'SzA', 'MD']
arguments_list_24 = gen_arguments(arg1, name, groups)


# ALL THE TUPLES
tuple_names = []
for i in range(1, 25):
    tuple_names.append(f'arguments_list_{i}')

arguments_list = []
for name in tuple_names:
    arguments_list += globals()[name]



""" MAIN """

if __name__ == "__main__":

    for args in arguments_list:
        command = ['python', '-m', 'additional.5_summary_differences'] + list(args)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()

        if process.returncode == 0:
            print(f'Script executed successfully with arguments {args}.\nOutput: \n{out.decode()}')
        else:
            print(f'Error occurred while executing with arguments {args}.\nError: {err.decode()}')

