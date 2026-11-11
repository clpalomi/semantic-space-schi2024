# Additional analysis; check additional/README.md before running.
from pipeline.paths import TEXTS, METADATA, ADDITIONAL_RESULTS

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
from transformers import AutoTokenizer, AutoModelWithLMHead
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


""" LANGUAGE MODEL TO LOAD: GPT-2 """

# Load pre-trained German GPT-2 model and tokenizer
model_name = "dbmdz/german-gpt2"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelWithLMHead.from_pretrained(model_name)



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


# FILTER TOKENS
def filter_tokens_by_vocabulary(tokenizer_name, string_list):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    filtered_list = [word for word in string_list if word in tokenizer.vocab]
    return filtered_list


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


# COINCIDENCE OF WORD LISTS
def find_coincidence(list1, list2):
    # Modify list1 to eliminate leading blank spaces
    modified_list1 = [s.lstrip() for s in list1]
    # Find positions of coincidence in the same sequential order
    positions = []
    i = j = 0
    while i < len(modified_list1) and j < len(list2):
        if modified_list1[i] == list2[j]:
            positions.append((i, j))
            i += 1
            j += 1
        else:
            j += 1
    return positions


# REMOVE FROM LIST OF STRING ANY SINGLE CHAR FOLLOWED BY A DOT
def remove_char_dot(strings):
    removed_positions = []
    filtered_strings = []
    for i, string in enumerate(strings):
        if len(string) == 2 and string[0].isalnum() and string.endswith('.'):
            removed_positions.append(i)
        else:
            filtered_strings.append(string)
    return filtered_strings, removed_positions


# REMOVE FROM LIST TWO SEQUENTUAL STRINGS OF A CHAR AND A DOT
def remove_seq_char_dot(strings):
    removed_positions = []
    filtered_strings = []
    i = 0
    while i < len(strings):
        if len(strings[i]) == 1 and strings[i].isalnum() and i + 1 < len(strings) and strings[i + 1] == '.':
            removed_positions.extend([i, i + 1])
            i += 2
        else:
            filtered_strings.append(strings[i])
            i += 1
    return filtered_strings, removed_positions


# ENTROPY
def entropy(text):
    prob = prob_words(text)[0]
    H = []
    for x in prob:
        H.append(-1*(x*np.log2(x) + (1-x)*np.log2(1-x)))
    return H


# ENTROPY LIST
def entropy_list(long_text):
    text_new = divide_text(long_text)
    # extend all the entropy values
    entropy_list = []
    for piece in text_new:
        h = entropy(piece)
        entropy_list.extend(h)
    return entropy_list


# RETURN THE PROBABILITY OF EACH WORD IN TEXT
def prob_words(text):
    # Tokenize the input sequence
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
    # Generate the logits using the GPT-2 model
    with torch.no_grad():
        logits = model(input_ids)[0]
    # Apply softmax to calculate probabilities
    probabilities = torch.softmax(logits, dim=-1)
    # Retrieve the probabilities for each token position
    token_probabilities = probabilities[0]
    probs = []
    tokens = [] 
    for i, token in enumerate(input_ids[0]):
        probability = token_probabilities[i][input_ids[0][i]].item()
        probs.append(probability)
        tokens.append(tokenizer.decode(token.tolist()))
    return [probs, tokens]


# FILTER BY WORDS IN SPACY
def filter_by_vocab(text):
    max_length = 1000000  # Maximum chunk size
    # Split the text into smaller chunks
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    filtered_words = []
    for chunk in chunks:
        doc = nlp(chunk)
        filtered_words.extend([token.text for token in doc if token.text in nlp.vocab])    
    return filtered_words


# GET TAGS
def get_pos_tags(text):
    max_length = 1000000  # Maximum chunk size
    # Split the text into smaller chunks
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    pos_tags = []
    tokens   = []
    for chunk in chunks:
        doc = nlp(chunk)
        pos_tags.extend([token.pos_ for token in doc])
        tokens.extend([token.text for token in doc])
    return [pos_tags, tokens]


# GET NOUN PHRASES
def get_noun_phrases(text):
    max_length = 1000000  # Maximum chunk size
    # Split the text into smaller chunks
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    noun_phrases = []
    for chunk in chunks:
        doc = nlp(chunk)
        noun_phrases.extend([x.text for x in doc.noun_chunks])
    return noun_phrases


# PLOT AND TEST
def plot_with_ttest(df, x, y, path=False, name_file=False):
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
    #if path!=False:
     #   plt.savefig(path_results+name_file+'.eps', dpi=1200, bbox_inches='tight')
      #  plt.savefig(path_results+name_file+'.jpg', dpi=1200, bbox_inches='tight')
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




""" PERPLEXITY GERMAN NEW """
text_files = {'Item 1a': 'Ein einsamer Hund entdeckte ein köstliches Fleischstück in einem großen Wald. Gierig packte er es fest und wollte es mit niemandem teilen. Als er einen großen Fluss überqueren musste, sah er sein Spiegelbild im Wasser und knurrte, da er dachte, es sei ein anderer Hund. Das Fleisch fiel ihm dabei ins Wasser und schwamm davon. So lernte er, der Gier nicht zu vertrauen.',
              'Item 1b': 'Ein einsamer Hund entdeckte das köstliche Fleischstück in manchen großen Wäldern. Gierig packten sie ihn fest und wollten sie mit jemandem teilen. Als er die großen Flüsse überqueren musste, sahen sie ihr Spiegelbild im Wasser und knurrten, da er dachte, es sei der andere Hund. Ein Fleisch fiel ihr dabei ins Wasser und schwamm davon. So lernten sie, der Gier nicht zu vertrauen.'}
for key, value in text_files.items():
    # Split the original string into a list of strings
    value_list = value.split()
    # Update the dictionary with the list
    text_files[key] = value_list

# START-RUN
start_time = time.time()

results = []
for key, lst in text_files.items():
    ent = entropy_list(join_words(lst))
    results.append({'name': key, 'mean': np.mean(ent), 'std': np.std(ent),
                    'range_s': np.max(ent) - np.min(ent),
                    'per10': np.percentile(ent,10),
                    'per90': np.percentile(ent,90)})
# END-RUN
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60
print(f"The semsim calculations took {elapsed_minutes:.2f} minutes to execute.")
# Create DataFrame from results list
df = pd.DataFrame(results)
""" PERPLEXITY GERMAN """






""" ENTROPY ALL """
# START-RUN
start_time = time.time()

results = []
for key, lst in text_files.items():
    ent = entropy_list(join_words(lst))
    results.append({'name': key, 'mean': np.mean(ent), 'std': np.std(ent),
                    'range_s': np.max(ent) - np.min(ent),
                    'per10': np.percentile(ent,10),
                    'per90': np.percentile(ent,90)})

# END-RUN
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60
print(f"The semsim calculations took {elapsed_minutes:.2f} minutes to execute.")

# Create DataFrame from results list
df = pd.DataFrame(results)

# Extract 'name' and '_bild'
df['bild'] = df['name'].str.extract(r'_bild_(\d+)')
df['name'] = df['name'].str.extract(r'(.+)_bild_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df = df.merge(groups, on='name', how='left')
# save to pickle file
df.to_pickle(path_results+'df_perplexity_all.pkl')


# AVERAGE VALUES (1,2,4,6)
non_numeric_df = df.select_dtypes(exclude=['number'])
non_numeric_df = non_numeric_df.drop_duplicates()
numeric_mean_df = df.groupby('name').mean().reset_index()
df_perplexity_av = pd.merge(numeric_mean_df, non_numeric_df, on='name')
df_perplexity_av = df_perplexity_av[['name', 's_group', 'mean', 'std', 'range_s', 'per10', 'per90']]
df_perplexity_av = df_perplexity_av.drop_duplicates().reset_index(drop=True)
df_perplexity_av.to_pickle(path_results+'df_perplexity_av.pkl')

# VALUES FOR EACH TASK
grouped = df.groupby('bild')
grouped_dfs = {}
for name, group in grouped:
    grouped_dfs[name] = group.copy()

# FOR EACH TASK
df_perplexity_1 = grouped_dfs['1'].reset_index(drop=True)
df_perplexity_2 = grouped_dfs['2'].reset_index(drop=True)
df_perplexity_4 = grouped_dfs['4'].reset_index(drop=True)
df_perplexity_6 = grouped_dfs['6'].reset_index(drop=True)

# TO PICKLE
df_perplexity_1.to_pickle(path_results+'df_perplexity_1.pkl')
df_perplexity_2.to_pickle(path_results+'df_perplexity_2.pkl')
df_perplexity_4.to_pickle(path_results+'df_perplexity_4.pkl')
df_perplexity_6.to_pickle(path_results+'df_perplexity_6.pkl')




""" ENTROPY BY TAG """

# LOAD NLP SPACY
nlp = spacy.load("de_core_news_lg")

# START-RUN
start_time = time.time()

results = []
for key, lst in text_files.items():
    ex_join  = join_words(lst)
    ex_filt  = filter_by_vocab(ex_join)
    ex_join  = join_words(ex_filt)
    ex_div   = divide_text(ex_join)
    ex_prob  = []
    ex_token = []
    for piece in ex_div:
        ex_prob.extend(prob_words(piece)[0])
        ex_token.extend(prob_words(piece)[1])
    
    m_ex_token = [s.lstrip() for s in ex_token]
    ex_tags     = []
    ex_token_n  = []
    text_from_tokens = join_words(m_ex_token, space=True)
    ex_div   = divide_text(text_from_tokens)
    for piece in ex_div:
        ex_tags.extend(get_pos_tags(piece)[0])
        ex_token_n.extend(get_pos_tags(piece)[1])
    
    
    # GET NOUN PHRASES
    noun_p = get_noun_phrases(ex_div[0])
    
    # FIND AND REMOVE INCONSISTENCIES 
    [fs_n, rp_n] = remove_char_dot(ex_token_n)
    [fs_t, rp_t] = remove_seq_char_dot(m_ex_token)

    for pos in sorted(rp_n, reverse=True):
        del ex_token_n[pos]
        
    for pos in sorted(rp_n, reverse=True):
        del ex_tags[pos]

    for pos in sorted(rp_t, reverse=True):
        del m_ex_token[pos]
        
    for pos in sorted(rp_t, reverse=True):
        del ex_prob[pos]
    
    # NUMBER OF WORDS
    n_words = len(ex_tags)
    
    # PROBABILITIES FOR SPECIFIC TAGS
    noun_phrases = len(noun_p)
    prob_pronoun = [prob for prob, tag in zip(ex_prob, ex_tags) if tag == 'PRON']
    prob_det     = [prob for prob, tag in zip(ex_prob, ex_tags) if tag == 'DET']
    prob_punct   = [prob for prob, tag in zip(ex_prob, ex_tags) if tag == 'PUNCT']
    prob_content = [prob for prob, tag in zip(ex_prob, ex_tags) if tag in ['NOUN', 'VERB', 'ADJ', 'ADV']]
    prob_nopron  = [prob for prob, tag in zip(ex_prob, ex_tags) if tag != 'PRON']

    results.append({'name': key, 'n_words': n_words, 'n_noun_phrases': noun_phrases/n_words,
                    'n_pron': len(prob_pronoun)/n_words, 'n_det': len(prob_det)/n_words,
                    'n_punct': len(prob_punct)/n_words, 'n_content': len(prob_content)/n_words,
                    'mean_pron': np.mean(prob_pronoun), 'sd_pron': np.std(prob_pronoun),
                    'mean_det': np.mean(prob_det), 'sd_det': np.std(prob_det),
                    'mean_punct': np.mean(prob_punct), 'sd_punct': np.std(prob_punct),
                    'mean_content': np.mean(prob_content), 'sd_content': np.std(prob_content),
                    'mean_nopron': np.mean(prob_nopron), 'sd_nopron': np.std(prob_nopron)})

# END-RUN
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60
print(f"The semsim calculations took {elapsed_minutes:.2f} minutes to execute.")

# Create DataFrame from results list
df_type = pd.DataFrame(results)
# Extract 'name' and '_bild'
df_type['bild'] = df_type['name'].str.extract(r'_bild_(\d+)')
df_type['name'] = df_type['name'].str.extract(r'(.+)_bild_\d+')
# Merge the 'groups' DataFrame based on matching values in the 'name' column
df_type = df_type.merge(groups, on='name', how='left')
# save to pickle file
df_type.to_pickle(path_results+'df_perplexity_type_all.pkl')


# AVERAGE VALUES (1,2,4,6)
non_numeric_df = df_type.select_dtypes(exclude=['number'])
non_numeric_df = non_numeric_df.drop_duplicates()
numeric_mean_df = df_type.groupby('name').mean().reset_index()
df_perplexity_type_av = pd.merge(numeric_mean_df, non_numeric_df, on='name')
df_perplexity_type_av = df_perplexity_type_av[['name', 's_group', 'n_words', 
                                               'n_noun_phrases', 'n_pron', 'n_det', 'n_punct',
                                               'n_content', 'mean_pron', 'mean_det', 'mean_punct',
                                               'mean_content', 'mean_nopron']]
df_perplexity_type_av = df_perplexity_type_av.drop_duplicates().reset_index(drop=True)
df_perplexity_type_av.to_pickle(path_results+'df_perplexity_type_av.pkl')

# VALUES FOR EACH TASK
grouped = df_type.groupby('bild')
grouped_dfs = {}
for name, group in grouped:
    grouped_dfs[name] = group.copy()

# FOR EACH TASK
df_perplexity_type_1 = grouped_dfs['1'].reset_index(drop=True)
df_perplexity_type_2 = grouped_dfs['2'].reset_index(drop=True)
df_perplexity_type_4 = grouped_dfs['4'].reset_index(drop=True)
df_perplexity_type_6 = grouped_dfs['6'].reset_index(drop=True)

# TO PICKLE
df_perplexity_type_1.to_pickle(path_results+'df_perplexity_type_1.pkl')
df_perplexity_type_2.to_pickle(path_results+'df_perplexity_type_2.pkl')
df_perplexity_type_4.to_pickle(path_results+'df_perplexity_type_4.pkl')
df_perplexity_type_6.to_pickle(path_results+'df_perplexity_type_6.pkl')



