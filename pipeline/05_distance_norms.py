from pipeline.text_measures import read_files_sent, sentence_lengths, length_statistics

from pipeline.paths import TEXTS, METADATA, RESULTS

"""
@author: Claudio
"""

# IMPORT PACKAGES
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import fasttext.util


# PATH INPUT
path_input = TEXTS
path_demo = METADATA
path_results = RESULTS


""" FUNCTIONS"""

# JOIN SENTENCES OF TEXT_FILES
def join_sentences(sentences):
    joined_text = ' '.join(sentences)
    punctuation_marks = r"[.,:;!?]"
    joined_text = re.sub(punctuation_marks, '', joined_text)
    joined_text = re.sub(r"(\w+)'[\w]+",r'\1', joined_text)
    # Split the joined text into a list of words
    word_list = re.findall(r'\b\w+\b', joined_text)
    return word_list


# FILTER BY CONTENT WORDS
def filter_content(word_list):
    # Join the list of words into a string
    text = ' '.join(word_list)
    # Process the text with spaCy
    doc = nlp(text)
    # Define content POS tags
    content_pos_tags = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    # Extract content words (tokens that are not stop words or punctuation)
    content_words = [token.text for token in doc if not token.is_stop and not token.is_punct and token.pos_ in content_pos_tags]
    return content_words


# SEMANTIC SIMILARITY CONSECUTIVE WORDS FASTTEXT
def sem_sim_fasttext(text):
    similarities = []
    # Iterate through the words in the list
    for i in range(1, len(text)):
        # Get word embeddings for consecutive words
        word1_embedding = ft.get_word_vector(text[i-1])
        word2_embedding = ft.get_word_vector(text[i])
        # Calculate cosine similarity between word embeddings
        similarity = np.dot(word1_embedding, word2_embedding) / (np.linalg.norm(word1_embedding) * np.linalg.norm(word2_embedding))
        # Append the similarity score to the list
        if not np.isnan(similarity):
            similarities.append(similarity)
    return similarities


# EUCLIDEAN DISTANCE
def euclidean_distance_fasttext(text):
    distances = []
    # Iterate through the words in the list
    for i in range(1, len(text)):
        # Get word embeddings for consecutive words
        word1_embedding = ft.get_word_vector(text[i-1])
        word2_embedding = ft.get_word_vector(text[i])
        # Calculate Euclidean distance between word embeddings
        distance = np.linalg.norm(word1_embedding - word2_embedding)
        # Append the distance score to the list
        if not np.isnan(distance):
            distances.append(distance)
    return distances


# DISTANCE NORM N
def norm_distance_fasttext(text, n=2):
    distances = []
    # Iterate through the words in the list
    for i in range(1, len(text)):
        # Get word embeddings for consecutive words
        word1_embedding = ft.get_word_vector(text[i-1])
        word2_embedding = ft.get_word_vector(text[i])
        # Calculate distance between word embeddings using the n-norm
        if n == np.inf:
            distance = np.max(np.abs(word1_embedding - word2_embedding))
        else:
            distance = np.linalg.norm(word1_embedding - word2_embedding, ord=n)
        # Append the distance score to the list
        if not np.isnan(distance):
            distances.append(distance)
    return distances


# NATURAL LOG
def natural_logarithm(values):
    # Convert the list to a NumPy array
    values_array = np.array(values)
    # Calculate the natural logarithm for each element
    log_values = np.log(values_array)
    return log_values


# SCATTER
def plot_scatter(values):
    # Generate x values (indices) corresponding to each y value in the list
    x_values = range(len(values))
    y_values = values
    # Create the scatter plot
    plt.scatter(x_values, y_values)
    # Add labels and title (optional)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Scatter Plot of List Values')
    # Display the plot
    plt.show()


""" GERMAN PROCESSING: SPLIT SPEECH INTO SENTENCES """

# SAVE EACH LIST OF WORDS
text_files = read_files_sent(path_input)
text_files = {key[:-4]: value for key, value in text_files.items()}
# IMPORT DEMOGRAPHICS
groups = pd.read_excel(path_demo+"groups.xlsx")
# UNIQUE NAMES
names = groups['name'].unique()
# Filter the 'text_files' dictionary based on the values in 'names'
text_files = {key: value for key, value in text_files.items() if key[:-7] in names}


""" LENGTH STATISTICS """

sent_lengths = sentence_lengths(text_files)
stats_length = length_statistics(sent_lengths)

# Extract 'name' and '_bild'
stats_length['bild'] = stats_length['filename'].str.extract(r'_bild_(\d+)')
stats_length['name'] = stats_length['filename'].str.extract(r'(.+)_bild_\d+')

# Merge the 'groups' DataFrame based on matching values in the 'name' column
stats_length = stats_length.merge(groups, on='name', how='left')

# stats to excel and pickle file
stats_length = stats_length[['name', 'bild', 'group', 'total_length',
                            'sent_av_length', 'sent_std_length']]


""" GERMAN - SEMANTIC SIMILARITTY BETWEEN WORDS """

# LOAD NLP SPACY
nlp = spacy.load("de_core_news_lg")

# LOAD FASTTEXT
fasttext.util.download_model('de', if_exists='ignore')  # German
ft = fasttext.load_model('cc.de.300.bin')

# LOOP FOR EACH SAMPLE
results   = []
centroids = []
for key, lst in text_files.items():
    print(key)
    joined_text  = ' '.join(lst)
    joined_list  = join_sentences(lst)
    c_words      = filter_content(joined_list)
    # semsim fasttext
    sem_list     = sem_sim_fasttext(c_words)
    sem_list     = [x for x in sem_list if x < 0.9999]
    log_s_list   = natural_logarithm(sem_list)
    n_content    = len(sem_list)+1
    # euclidean distance fasttext
    dist_list    = euclidean_distance_fasttext(c_words)
    dist_list    = [x for x in dist_list if x > 0.0001]
    log_list     = natural_logarithm(dist_list)
    n_content_d  = len(dist_list)+1
    # dist norm 1
    dist_1_list  = norm_distance_fasttext(c_words, n=1)
    dist_1_list  = [x for x in dist_1_list if x > 0.0001]
    log_1_list   = natural_logarithm(dist_1_list)
    # dist norm 3
    dist_3_list  = norm_distance_fasttext(c_words, n=3)
    dist_3_list  = [x for x in dist_3_list if x > 0.0001]
    log_3_list   = natural_logarithm(dist_3_list)
    # dist norm 4
    dist_4_list  = norm_distance_fasttext(c_words, n=4)
    dist_4_list  = [x for x in dist_4_list if x > 0.0001]
    log_4_list   = natural_logarithm(dist_4_list)
    # dist norm 5
    dist_5_list  = norm_distance_fasttext(c_words, n=5)
    dist_5_list  = [x for x in dist_5_list if x > 0.0001]
    log_5_list   = natural_logarithm(dist_5_list)
    # Chebyshev
    dist_inf_list  = norm_distance_fasttext(c_words, n=np.inf)
    dist_inf_list  = [x for x in dist_inf_list if x > 0.0001]
    log_inf_list   = natural_logarithm(dist_inf_list)
    # MEAN VALUES
    mean_semsim      = np.mean(sem_list).astype(float)
    mean_log_s       = np.mean(log_s_list).astype(float)
    mean_dist        = np.mean(dist_list).astype(float)
    mean_log_dist    = np.mean(log_list).astype(float)
    mean_dist_1      = np.mean(dist_1_list).astype(float)
    mean_log_1_dist  = np.mean(log_1_list).astype(float)
    mean_dist_3      = np.mean(dist_3_list).astype(float)
    mean_log_3_dist  = np.mean(log_3_list).astype(float)
    mean_dist_4      = np.mean(dist_4_list).astype(float)
    mean_log_4_dist  = np.mean(log_4_list).astype(float)
    mean_dist_5      = np.mean(dist_5_list).astype(float)
    mean_log_5_dist  = np.mean(log_5_list).astype(float)
    mean_dist_i      = np.mean(dist_inf_list).astype(float)
    mean_log_i_dist  = np.mean(log_inf_list).astype(float)
    results.append({'name': key, 'n_content': n_content, 'n_content_d': n_content_d,
                    'mean_semsim': mean_semsim,
                    'mean_log_s': mean_log_s,
                    'mean_dist': mean_dist,
                    'mean_log_dist': mean_log_dist,
                    'mean_dist_1': mean_dist_1,
                    'mean_log_1_dist': mean_log_1_dist,
                    'mean_dist_3': mean_dist_3,
                    'mean_log_3_dist': mean_log_3_dist,
                    'mean_dist_4': mean_dist_4,
                    'mean_log_4_dist': mean_log_4_dist,
                    'mean_dist_5': mean_dist_5,
                    'mean_log_5_dist': mean_log_5_dist,
                    'mean_dist_i': mean_dist_i,
                    'mean_log_i_dist': mean_log_i_dist
                    })

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
# merge
df = pd.merge(df, stats_length[['name', 'group', 'bild', 'total_length', 'sent_av_length']], on=['name', 'group', 'bild'], how='left')
column_data = df.pop('total_length')
df.insert(3, 'total_length', column_data)
column_data = df.pop('sent_av_length')
df.insert(4, 'sent_av_length', column_data)
# save to pickle file
df.to_pickle(path_results+'df_distances_norms.pkl')

