# Additional analysis; check additional/README.md before running.
from pipeline.paths import TEXTS, METADATA, ADDITIONAL_RESULTS

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 04:21:34 2023

@author: Claudio
"""
# INSTALL

# pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.6.0/de_core_news_lg-3.6.0-py3-none-any.whl
# pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.6.0/de_core_news_lg-3.6.0.tar.gz (THIS ONE)


# IMPORT PACKAGES
import os
import re
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import time
import seaborn as sns
import spacy


# PATH INPUT
path_input = TEXTS
path_demo = METADATA
path_results = ADDITIONAL_RESULTS


""" LANGUAGE MODEL TO LOAD: BERT """

# Load pre-trained German GPT-2 model and tokenizer
model_name = "dbmdz/bert-base-german-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)



""" FUNCTIONS"""

# CREATE A DICTIONARY OF TEXT FROM FOLDER (different from semsim_sz_1)
def read_files(folder_path):
    files_dict = {}
    pattern = r'^.*bild_\d+'
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if re.search(pattern, filename) and filename.endswith(".txt"):
                print(filename)
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    words = re.findall(r'\w+|[^\w\s]', text)  # Split text into words and treat punctuation marks as separate values
                    files_dict[filename] = words
    return files_dict


# TO JOIN WORDS IN A LIST
def join_words(word_list, space=True):
    punctuation_marks = r"[.,:;!?]"
    if space==False:
        text = ''.join(word_list)
        text = re.sub(f"\s*({punctuation_marks})\s*", r"\1 ", text)
    if space==True:
        text = ' '.join(word_list)
        text = re.sub(f"\s*({punctuation_marks})\s*", r"\1 ", text)
    return text


# FILTER BY CONTENT WORDS
def filter_content(text):
    # Chunk the text into smaller pieces with a maximum length
    max_length = 1000000
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    # Initialize an empty list to store the content words
    content_words = []
    # Process each chunk with spaCy
    for chunk in chunks:
        doc = nlp(chunk)
        # Extract content words (tokens that are not stop words or punctuation)
        content_words.extend(token.text for token in doc if not token.is_stop and not token.is_punct)
    # Join the content words to form the final result
    result = ' '.join(content_words)
    return result


# DIVIDE A TEXT INTO PIECES
def divide_text(text):
    words = text.split()
    num_words = len(words)
    max_words_per_piece = 500
    num_pieces = (num_words + max_words_per_piece - 1) // max_words_per_piece
    divided_text = []
    for i in range(num_pieces):
        start_index = i * max_words_per_piece
        end_index = (i + 1) * max_words_per_piece
        piece = ' '.join(words[start_index:end_index])
        divided_text.append(piece)
    return divided_text


# SEMANTIC SIMILARITY IN CHUNKS
def sem_sim(text):
    max_tokens = 500
    #tokens = tokenizer.tokenize(text)
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")['input_ids']
    num_tokens = len(tokens[0])
    if num_tokens <= max_tokens:
        return process_chunk(tokens)
    results = []
    for i in range(0, num_tokens, max_tokens):
        chunk = tokens[i:i+max_tokens]
        results.extend(process_chunk(chunk))
    return results


# SEMANTIC SIMILARITY IN CHUNKS
def process_chunk(tokens):
    #token_ids = tokenizer.convert_tokens_to_ids(tokens)
    #token_ids = torch.tensor(token_ids).unsqueeze(0)
    # Get word embeddings from BERT
    with torch.no_grad():
        #outputs = model(token_ids)
        outputs = model(tokens)
        #embeddings = outputs.hidden_states[-1].mean(dim=1).squeeze()
        embeddings = outputs.hidden_states[-1]
        embeddings = embeddings[0]
        #embeddings = outputs[0]   # This contains word embeddings
    # Calculate cosine similarity between consecutive word embeddings
    #word_embeddings = embeddings.squeeze(0)  # Remove batch dimension
    # Initialize a list to store the cosine similarities
    similarities = []
    # Iterate through the embeddings
    for i in range(1, embeddings.shape[0]):
        # Get consecutive word embeddings
        word1 = embeddings[i-1]
        word2 = embeddings[i]        
        # Reshape to make sure they are of shape (embedding_dim,)
        word1 = word1.view(-1)
        word2 = word2.view(-1)
        # Calculate cosine similarity
        similarity = F.cosine_similarity(word1, word2, dim=0)
        # Append to the list of similarities
        similarities.append(similarity.item())
    #similarities = cosine_similarity(word_embeddings, word_embeddings)
    # Extract diagonal values to get cosine similarity between consecutive words
    # cosine_similarities = [similarities[i, i+1] for i in range(len(tokens) - 1)]
    return similarities


# BOX-PLOT WIHT TTEST (REGULAR)
def plot_with_ttest_1(df, x, y, path=False, name_file=False):
    # GRAPHS AND DIFFERENCES
    plt.boxplot([df[df[x] == val][y].dropna() for val in df[x].unique()], showfliers=False)
    plt.xticks(range(1, len(df[x].unique()) + 1), df[x].unique())
    # Perform t-test
    groups = [df[df[x] == val][y].dropna() for val in df[x].unique()]
    t_statistic, p_value = stats.ttest_ind(*groups)
    # Add significance indicator
    if p_value < 0.05:
        significance_indicator = '*'
    else:
        significance_indicator = 'ns'
    # Add p-value to the plot
    plt.text(0.5, 1.05, f'p-value: {p_value:.4f} {significance_indicator}',
             horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes)
    # Increase the size of the axis titles
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(x, fontdict={'fontsize': 14})
    plt.ylabel(y, fontdict={'fontsize': 14})
    if path != False:
        plt.savefig(path_results + name_file + '.eps', dpi=1200, bbox_inches='tight')
        plt.savefig(path_results + name_file + '.jpg', dpi=1200, bbox_inches='tight')
    plt.show()


#  BOX-PLOT WIHT TTEST (SEABORN)
def plot_with_ttest_2(df, x, y, path=False, name_file=False):
    # GRAPHS AND DIFFERENCES
    sns.boxplot(x=x, y=y, data=df, showfliers=False)
    # Perform t-test
    group1 = df[df[x] == df[x].unique()[0]][y]
    group2 = df[df[x] == df[x].unique()[1]][y]
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
    if path!=False:
        plt.savefig(path_results+name_file+'.eps', dpi=1200, bbox_inches='tight')
        plt.savefig(path_results+name_file+'.jpg', dpi=1200, bbox_inches='tight')
    plt.show()



""" PROCESSING """

# SAVE EACH LIST OF WORDS
text_files = read_files(path_input)
text_files = {key[:-4]: value for key, value in text_files.items()}

# IMPORT DEMOGRAPHICS
groups = pd.read_excel(path_demo+"groups.xlsx")

# UNIQUE NAMES
names = groups['name'].unique()

# Filter the 'text_files' dictionary based on the values in 'names'
text_files = {key: value for key, value in text_files.items() if key[:-7] in names}



""" SEMANTIC SIMILARITY """

# LOAD NLP SPACY
nlp = spacy.load("de_core_news_lg")

# START-RUN
start_time = time.time()

results = []
for key, lst in text_files.items():
    joined         = join_words(lst)
    l_words        = joined.split(' ')
    length_s       = len(l_words)
    filtered_text  = filter_content(joined)
    c_words        = filtered_text.split(' ')
    n_content      = len(c_words)
    sem_list       = sem_sim(filtered_text)
    results.append({'name': key, 'mean': np.mean(sem_list), 'std': np.std(sem_list),
                    'length': length_s, 'n_content': n_content,
                    'range_s': np.max(sem_list) - np.min(sem_list),
                    'per10': np.percentile(sem_list,10),
                    'per90': np.percentile(sem_list,90)})

# END-RUN
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60
print(f"The semsim calculations took {elapsed_minutes:.2f} minutes to execute.")



""" NEW """
sns.histplot(data, bins=range(min(data), max(data) + 1), kde=False, color='skyblue', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Values')
plt.show()

""" """


# Create DataFrame from results list
df = pd.DataFrame(results)

# Extract 'name' and '_bild'
df['bild'] = df['name'].str.extract(r'_bild_(\d+)')
df['name'] = df['name'].str.extract(r'(.+)_bild_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df = df.merge(groups, on='name', how='left')
# save to pickle file
df.to_pickle(path_results+'df_semsim_all.pkl')


# AVERAGE VALUES (1,2,4,6)
non_numeric_df = df.select_dtypes(exclude=['number'])
non_numeric_df = non_numeric_df.drop_duplicates()
numeric_mean_df = df.groupby('name').mean().reset_index()
df_semsim_av = pd.merge(numeric_mean_df, non_numeric_df, on='name')
df_semsim_av = df_semsim_av[['name', 's_group', 'mean', 'std', 'range_s', 'per10', 'per90']]
df_semsim_av = df_semsim_av.drop_duplicates().reset_index(drop=True)
df_semsim_av.to_pickle(path_results+'df_semsim_av.pkl')

# VALUES FOR EACH TASK
grouped = df.groupby('bild')
grouped_dfs = {}
for name, group in grouped:
    grouped_dfs[name] = group.copy()

# FOR EACH TASK
df_semsim_1 = grouped_dfs['1'].reset_index(drop=True)
df_semsim_2 = grouped_dfs['2'].reset_index(drop=True)
df_semsim_4 = grouped_dfs['4'].reset_index(drop=True)
df_semsim_6 = grouped_dfs['6'].reset_index(drop=True)

# TO PICKLE
df_semsim_1.to_pickle(path_results+'df_semsim_1.pkl')
df_semsim_2.to_pickle(path_results+'df_semsim_2.pkl')
df_semsim_4.to_pickle(path_results+'df_semsim_4.pkl')
df_semsim_6.to_pickle(path_results+'df_semsim_6.pkl')



