from pipeline.text_measures import read_files_sent

# Additional analysis; check additional/README.md before running.
from pipeline.paths import TEXTS, METADATA, SANTANDER, ADDITIONAL_RESULTS

"""
@author: Claudio
"""

import os
import re
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import fasttext.util
import spacy
from openTSNE import TSNE
from sentence_transformers import SentenceTransformer
from scipy.spatial import ConvexHull

# PATHS
path_input = TEXTS
path_demo = METADATA
path_santander = SANTANDER
path_results = ADDITIONAL_RESULTS


""" FUNCTIONS """


# CREATE A DICTIONARY OF TEXT FROM FOLDER


# JOIN SENTENCES OF TEXT_FILES
def join_sentences(sentences):
    joined_text = ' '.join(sentences)
    punctuation_marks = r"[.,:;!?]"
    joined_text = re.sub(punctuation_marks, '', joined_text)
    # Split the joined text into a list of words
    word_list = re.findall(r'\b\w+\b', joined_text)
    return word_list


# FILTER BY CONTENT WORDS
def filter_content(word_list):
    # Join the list of words into a string
    text = ' '.join(word_list)
    # Process the text with spaCy
    doc = nlp(text)
    # Extract content words (tokens that are not stop words or punctuation)
    content_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return content_words


# REDUCE DIMENSIONALITY SENTENCES
def reduce_dim(sentences, dim=5):
    sentence_embeddings = model.encode(sentences)
    tsne = TSNE(n_components=dim, perplexity=4, n_jobs=-1, random_state=42)
    tsne_embeddings = tsne.fit(sentence_embeddings)
    tsne_embeddings = np.array(tsne_embeddings)
    return tsne_embeddings


# REDUCE DIMENSIONALITY WORDS
def reduce_dim_words(vectors_words, dim=5):
    tsne = TSNE(n_components=dim, perplexity=4, n_jobs=-1, random_state=42)
    tsne_embeddings = tsne.fit(vectors_words)
    tsne_embeddings = np.array(tsne_embeddings)
    return tsne_embeddings



""" READ GERMAN DATA: SPLIT SPEECH INTO SENTENCES """

# SAVE EACH LIST OF WORDS
text_german = read_files_sent(path_input)
text_german = {key[:-4]: value for key, value in text_german.items()}
# IMPORT DEMOGRAPHICS
groups = pd.read_excel(path_demo+"groups.xlsx")
# UNIQUE NAMES
names_german = groups['name'].unique()
# Filter the 'text_files' dictionary based on the values in 'names'
text_german = {key: value for key, value in text_german.items() if key[:-7] in names_german}



""" t-SNE EMBEDDINGS CALCULATIONS """


""" SENTENCES """

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

results = []
for key, lst in text_german.items():
    print(key)
    n_sent  = len(lst)
    if n_sent <=5:
        area   = 0
        volume = 0
    else:
        points = reduce_dim(lst)
        convex_hull  = ConvexHull(points)
        area   = convex_hull.area
        volume = convex_hull.volume
    results.append({'name':  key, 'n_sent': n_sent,
                    'area': area, 'volume': volume
                    })


# RESULTS DATAFRAME
df = pd.DataFrame(results)
# Extract 'name' and '_bild'
df['bild'] = df['name'].str.extract(r'_bild_(\d+)')
df['name'] = df['name'].str.extract(r'(.+)_bild_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df = df.merge(groups, on='name', how='left')
# Remove columns
df = df.drop(columns=['s_group', 'type'])
# change positions
column_data = df.pop('group')
df.insert(1, 'group', column_data)
column_data = df.pop('bild')
df.insert(2, 'bild', column_data)
# save to pickle file
df.to_pickle(path_results+'df_sent_convexhull.pkl')
df = pd.read_pickle(path_results+'df_sent_convexhull.pkl')


# FILTER AND NORMALIZATION
df = df[df['n_sent'] > 5]

df_1 = df[df['bild']=='1']
df_2 = df[df['bild']=='2']
df_4 = df[df['bild']=='4']
df_6 = df[df['bild']=='6']



""" WORDS (FASTTEXT) """


# LOAD NLP SPACY
nlp = spacy.load("de_core_news_lg")

# LOAD FASTTEXT
fasttext.util.download_model('de', if_exists='ignore')  # German
ft = fasttext.load_model('cc.de.300.bin')

results = []
for key, lst in text_german.items():
    print(key)
    joined_list  = join_sentences(lst)
    c_words      = filter_content(joined_list)
    n_words      = len(c_words)
    if n_words <=5:
        area   = 0
        volume = 0
    else:
        word_vectors = [ft.get_word_vector(word) for word in c_words]
        word_vectors = np.array(word_vectors)
        points       = reduce_dim_words(word_vectors)
        convex_hull  = ConvexHull(points)
        area   = convex_hull.area
        volume = convex_hull.volume
    results.append({'name':  key, 'n_words': n_words,
                    'area': area, 'volume': volume
                    })

# RESULTS DATAFRAME
df_words = pd.DataFrame(results)



""" GRAPHS """

# SCATTER PLOTS
def scatter_plot_group(df, var1, var2, group=None, pict=None):
    # Filter the DataFrame based on 'group' and 'pict'
    if group is not None:
        df = df[df['group'] == group]
    if pict is not None:
        df = df[df['bild'] == pict]
    # Scatter plot with different colors for different groups
    sns.set_palette("colorblind")
    if group is None:
        grouped = df.groupby('group')
        for name, group_data in grouped:
            plt.scatter(group_data[var1], group_data[var2], label=f'{name}')
        plt.legend()
    else:
        plt.scatter(df[var1], df[var2])
    # Add labels and title
    plt.xlabel(f'{var1}')
    plt.ylabel(f'{var2}')
    plt.title(f'{var2} vs {var1}')
    #plt.savefig(path_results+'scatter_corr.jpg', dpi=500)
    plt.show()


# BOXPLOTS
def boxplot_group_diff(df, var1, group=None, pict=None):
    if pict is not None:
        df = df[df['bild'] == pict]
    # Exclude rows where var1 is 0
    df = df[df[var1] != 0]
    sns.set_palette("colorblind")
    sns.boxplot(x='group', y=var1, data=df)
    # Add labels and title
    plt.xlabel('Group')
    plt.ylabel(f'{var1}')
    plt.title(f'Boxplot of {var1} by Group and Picture')
    # Show the plot
    plt.show()




    
boxplot_group_diff(df, 'n_sent')
boxplot_group_diff(df, 'area')
boxplot_group_diff(df, 'volume')

boxplot_group_diff(df_1, 'n_sent')
boxplot_group_diff(df_1, 'area')
boxplot_group_diff(df_1, 'volume')

boxplot_group_diff(df_2, 'n_sent')
boxplot_group_diff(df_2, 'area')
boxplot_group_diff(df_2, 'volume')

boxplot_group_diff(df_4, 'n_sent')
boxplot_group_diff(df_4, 'area')
boxplot_group_diff(df_4, 'volume')

boxplot_group_diff(df_6, 'n_sent')
boxplot_group_diff(df_6, 'area')
boxplot_group_diff(df_6, 'volume')



## TTEST

from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu


# COMPARE GROUPS
def compare_groups(df, variable_column, group_column, group1, group2):
    group_a = df[df[group_column] == group1][variable_column]
    group_b = df[df[group_column] == group2][variable_column]
    t_statistic, p_value = ttest_ind(group_a, group_b)
    avg_group_a = group_a.mean()
    avg_group_b = group_b.mean()
    result = {
        't_statistic': t_statistic,
        'p_value': p_value,
        'avg_{}_{}'.format(group_column, group1): avg_group_a,
        'avg_{}_{}'.format(group_column, group2): avg_group_b,
    }
    if avg_group_a > avg_group_b:
        result['higher_avg'] = group1
    elif avg_group_b > avg_group_a:
        result['higher_avg'] = group2
    else:
        result['higher_avg'] = 'No difference'
    return result


# COMPARE GROUPS with Mann-Whitney U test
def compare_groups_nonparametric(df, variable_column, group_column, group1, group2):
    group_a = df[df[group_column] == group1][variable_column]
    group_b = df[df[group_column] == group2][variable_column]
    statistic, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')
    avg_group_a = group_a.median()  # Using median since mean may not be meaningful in non-parametric tests
    avg_group_b = group_b.median()
    result = {
        'statistic': statistic,
        'p_value': p_value,
        'avg_{}_{}'.format(group_column, group1): avg_group_a,
        'avg_{}_{}'.format(group_column, group2): avg_group_b,
    }
    if avg_group_a > avg_group_b:
        result['higher_avg'] = group1
    elif avg_group_b > avg_group_a:
        result['higher_avg'] = group2
    else:
        result['higher_avg'] = 'No difference'
    return result


compare_groups(df, 'volume', 'group', 'Sz', 'HC')   # SIG
compare_groups(df, 'volume', 'group', 'Sz', 'MD')
compare_groups(df, 'volume', 'group', 'MD', 'HC')

compare_groups_nonparametric(df, 'volume', 'group', 'Sz', 'HC')
compare_groups_nonparametric(df, 'volume', 'group', 'Sz', 'MD')
compare_groups_nonparametric(df, 'volume', 'group', 'MD', 'HC')

compare_groups(df, 'volume', 'bild', '1', '2')  # SIG <
compare_groups(df, 'volume', 'bild', '1', '4')
compare_groups(df, 'volume', 'bild', '1', '6')  # SIG <
compare_groups(df, 'volume', 'bild', '2', '4')
compare_groups(df, 'volume', 'bild', '2', '6')
compare_groups(df, 'volume', 'bild', '4', '6')



compare_groups(df, 'area', 'group', 'Sz', 'HC')
compare_groups(df, 'area', 'group', 'Sz', 'MD')
compare_groups(df, 'area', 'group', 'MD', 'HC')

compare_groups_nonparametric(df, 'area', 'group', 'Sz', 'HC')
compare_groups_nonparametric(df, 'area', 'group', 'Sz', 'MD')
compare_groups_nonparametric(df, 'area', 'group', 'MD', 'HC')

compare_groups_nonparametric(df, 'volume', 'bild', '1', '2')  # SIG >
compare_groups_nonparametric(df, 'volume', 'bild', '1', '4')  # SIG >
compare_groups_nonparametric(df, 'volume', 'bild', '1', '6')  # SIG >
compare_groups_nonparametric(df, 'volume', 'bild', '2', '4')
compare_groups_nonparametric(df, 'volume', 'bild', '2', '6')
compare_groups_nonparametric(df, 'volume', 'bild', '4', '6')



## MIXED MODEL

from statsmodels.regression.mixed_linear_model import MixedLM

def mixed_model_results(df, y):
    model_r = MixedLM.from_formula(f'{y} ~ group + bild + n_sent', df, groups=df["name"])
    result  = model_r.fit()
    print(result.summary())
    
mixed_model_results(df, 'volume')    
mixed_model_results(df, 'area')
