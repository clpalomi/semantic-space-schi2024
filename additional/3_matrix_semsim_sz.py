# Additional analysis; check additional/README.md before running.
from pipeline.paths import TEXTS, METADATA, NETWORK_RESULTS

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
import numpy as np
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import time
import seaborn as sns
import spacy
import umap

# PATH INPUT
path_input = TEXTS
path_demo = METADATA
path_results = NETWORK_RESULTS


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
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")['input_ids']
    num_tokens = len(tokens[0])
    if num_tokens <= max_tokens:
        return process_chunk(tokens)
    results = []
    for i in range(0, num_tokens, max_tokens):
        chunk = tokens[i:i+max_tokens]
        results.extend(process_chunk(chunk))
    return results


# MATRIX OF COSINE SIMILARITIES
def matrix_cos_similarity(matrix):
    norm = torch.norm(matrix, dim=1, keepdim=True)
    normalized_matrix = matrix / norm
    similarity_matrix = torch.matmul(normalized_matrix, normalized_matrix.T)
    return similarity_matrix


# SEMANTIC SIMILARITY IN CHUNKS
def process_chunk(tokens):
    with torch.no_grad():
        outputs = model(tokens)
        embeddings = outputs.hidden_states[-1]
        embeddings = embeddings[0][1:-1]   # remove initial and final token
        # Create a matrix with all pairwise cosine similarities
    sim_matrix = matrix_cos_similarity(embeddings)
    sim_matrix = np.round(sim_matrix, 6)
    sim_matrix = pd.DataFrame(sim_matrix.numpy())
    return sim_matrix


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


# GET THE GROUP
def get_group(string):
    # Assuming `groups` is a pandas DataFrame with columns 'name' and 'group'
    result = groups[groups['name'].str[:] == string[:-7]]['group'].values
    if len(result) > 0:
        return result[0]
    else:
        return None


# GET S-GROUP
def get_sgroup(string):
    # Assuming `groups` is a pandas DataFrame with columns 'name' and 'group'
    result = groups[groups['name'].str[:] == string[:-7]]['s_group'].values
    if len(result) > 0:
        return result[0]
    else:
        return None


# MATRIX SIMILARITY TO AVERAGE NUMBERS
def mat_sim_measures(mat_sim):
    largo = len(mat_sim)
    arr   = mat_sim.to_numpy()
    arr   = [abs(x) for x in list(arr[np.triu_indices(largo)]) if x<1]
    return [np.mean(arr), np.std(arr), largo]


# MATRIX SIMILARITY DISTRIBUTION
def mat_sim_dist(mat_sim):
    largo = len(mat_sim)
    arr   = mat_sim.to_numpy()
    arr   = [abs(x) for x in list(arr[np.triu_indices(largo)]) if x<1]
    return arr




""" PROCESSING """

# SAVE EACH LIST OF WORDS
text_files = read_files(path_input)
text_files = {key[:-4]: value for key, value in text_files.items()}
# IMPORT DEMOGRAPHICS
groups = pd.read_excel(path_demo+"groups.xlsx")
# UNIQUE NAMES
names = groups['name'].unique()
# Filter the 'text_files' dictionary based on the values in 'names'
text_files  = {key: value for key, value in text_files.items() if key[:-7] in names}




""" MATRIX OF SEMANTIC SIMILARITY """

# START-RUN
start_time = time.time()
# LOAD NLP SPACY
nlp = spacy.load("de_core_news_lg")
all_matrix_sim = {}
for key, lst in text_files.items():
    joined         = join_words(lst)
    filtered_text  = filter_content(joined)
    sim_matrix     = sem_sim(filtered_text)
    all_matrix_sim[key] = sim_matrix
# END-RUN
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60
print(f"The semsim calculations took {elapsed_minutes:.2f} minutes to execute.")




""" FUNCTIONS FOR GRAPH SEM SIM DISTANCES """
import networkx as nx

# PLOT UMAP
def plot_umap(sim_matrix_data, name, group, rs=1):
    data_u = sim_matrix_data.astype(float)
    # Perform UMAP embedding
    reducer = umap.UMAP(n_components=2, random_state=rs)
    embedded_points = reducer.fit_transform(data_u)
    # Create a DataFrame to hold the embedded points
    embedded_df = pd.DataFrame(data=embedded_points, columns=['X', 'Y'], index=data_u.index)
    # Visualize the embedded points
    plt.figure(figsize=(8, 8))
    plt.scatter(embedded_df['X'], embedded_df['Y'])
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('UMAP Embedding of '+name+ ' - ' + group)
    #plt.savefig(path_results+'umap_'+name+'_'+group+'.jpg', dpi=1000, bbox_inches='tight')
    plt.show()


# GET EMBEDDING UMAP
def get_umap(sim_matrix_data, rs=1):
    data_u = sim_matrix_data.astype(float)
    # Perform UMAP embedding
    reducer = umap.UMAP(n_components=2, random_state=1)
    embedded_points = reducer.fit_transform(data_u)
    # Create a DataFrame to hold the embedded points
    embedded_df = pd.DataFrame(data=embedded_points, columns=['X', 'Y'], index=data_u.index)
    return embedded_df


# GENERATE EDGES
def generate_edges(sim_matrix_data):
    edges = []
    num_nodes = len(sim_matrix_data)
    for i in range(num_nodes - 1):
        edges.append((i, i+1))
    return edges


# GENERATE EDGES WITH THRESHOLD
def generate_edges_t(sim_matrix_data, threshold):
    edges = []
    num_nodes = len(sim_matrix_data)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if sim_matrix_data[i][j] > threshold:
                edges.append((i, j))
    return edges


# ELIMINATE VALUES
def survive_values(arr):
    arr = [x for x in arr if x != 0]
    if len(arr) == 1:
        return arr[0]
    elif len(arr) > 1:
        return max(arr)
    else:
        return None


# SMALL-WORLDNESS INDEX
def small_world(embedded_df, edges):
    # Graph G
    G = nx.Graph()
    for idx, row in embedded_df.iterrows():
        G.add_node(idx, x=row['X'], y=row['Y'])
    G.add_edges_from(edges)
    # Graph random
    G_random = nx.gnm_random_graph(n=len(embedded_df), m=G.number_of_edges(), seed=86)
    # Connected components G
    connected_components = list(nx.connected_components(G))
    shortest_G = []
    for component in connected_components:
        subgraph = G.subgraph(component)
        av_path_length = nx.average_shortest_path_length(subgraph)
        shortest_G.append(av_path_length)
    # Connected components G_random
    connected_components = list(nx.connected_components(G_random))
    shortest_Gr = []
    for component in connected_components:
        subgraph = G_random.subgraph(component)
        av_path_length = nx.average_shortest_path_length(subgraph)
        shortest_Gr.append(av_path_length)    
    # Calculate properties
    #L_real = nx.average_shortest_path_length(G)
    L_real = survive_values(shortest_G)
    C_real = nx.average_clustering(G)
    L_rand = survive_values(shortest_Gr)
    C_rand = nx.average_clustering(G_random)
    if (C_rand!=0 and L_real!=0 and L_rand!=0):
        sigma = (C_real / C_rand) / (L_real / L_rand)
    else:
        sigma = np.nan
    return sigma


# RETURNS CENTRALITY (AVERAGE) AND CLUSTERING COEFFICIENT
def centrality(embedded_df, edges):
    G = nx.Graph()
    for idx, row in embedded_df.iterrows():
        G.add_node(idx, x=row['X'], y=row['Y'])
    G.add_edges_from(edges)
    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(G)
    av_centrality = sum(degree_centrality.values()) / len(degree_centrality)
    # Calculate clustering coefficient
    clustering_coefficient = nx.average_clustering(G)
    return [av_centrality, clustering_coefficient]


# RETURNS THRESHOLDED MATRIX
def threshold_proportional(W, p, copy=True):
    """
    brainconn downloaded from https://github.com/fiuneuro/brainconn
    Modified from the "matrix.py" script from brainconn:
    1. We set all negative values to zeros too before thresholding
    2. We threshold the top p percentage of coefficients per row, not in the 
    whole matrix
    3. We used np.ceil instead of np.round for thresholding numbers to avoid
    singular matrix which is harmful for later analysis
    4. If there are more than one smallest values in thresholding (e.g. in a 
    row where the smallest value to enter threshold is 0.3 but there are three
    0.3 in the row), all smallest values enter the threshold. In the original 
    funtion, only the last one entered.
    This function "thresholds" the connectivity matrix by preserving a
    proportion p (0<p<1) of the strongest weights. All other weights are \
    set to 0.
    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : :obj:`numpy.ndarray`
        weighted connectivity matrix
    p : float
        proportional weight threshold (0<p<1)
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : :obj:`numpy.ndarray`
        thresholded connectivity matrix
    """
    if p > 1 or p < 0:
        raise ValueError('Threshold must be value between 0 and 1')
    if copy:
        W = W.to_numpy()
        W = W.copy()
        W = np.triu(W, k=1) + np.triu(W, k=1).T
    # number of nodes
    n = len(W)		
    # clear diagonal				
    np.fill_diagonal(W, 0)
    # ensure symmetry is preserved
    if np.allclose(W, W.T):				
        W[np.tril_indices(n)] = 0
        # halve number of removed links	
        ud = 2			
    else:
        ud = 1
    W[W<0]=0    	
    # thresholding  
    W_thresholded = np.zeros(W.shape)
    for i in range(W.shape[0]):
        W_unit = W[i, :]
        #  find all links
        ind = np.where(W_unit)[0]
        # sort indices by magnitude
        sorted_values = np.argsort(W_unit[ind])[::-1]
        # number of links to be preserved
        en = int(np.ceil(((n - i - 1) * p) / ud))    
        # values to maintain
        ind_thre = ind[sorted_values][:en+1]
        # find if there are any same value 
        if ind_thre.shape[0] != 0:
            for t in ind[sorted_values][en:]:
                if W_unit[ind_thre[-1]] == W_unit[t]:
                    ind_thre = np.append(ind_thre, t)
        # apply threshold
        W_unit[ind[np.isin(ind, ind_thre, invert=True)]] = 0
        W_thresholded[i, :] = W_unit
    # # if symmetric matrix
    if ud == 2:
        W[:, :] = W_thresholded + W_thresholded.T	
    else:
        W[:, :] = W
    return W


# SELECT A THRESHOLD
def select_threshold(affinity_matrix, t=0.55):
    '''
    Threshold reference:
    Watts, D., Strogatz, S. Collective dynamics of ‘small-world’ networks. 
    Nature 393, 440–442 (1998). https://doi.org/10.1038/30918
    We DO NOT apply the 1.1 as minimum sigma as this is a value derived from
    https://www.sciencedirect.com/science/article/pii/S0006322311005476#sec5,
    i.e. from their spefcific data, to control the threshold lower than 0.34, 
    which means it's NOT necessarily applicable to all data, e.g. the London 
    speech data. We choose a range of threshold from 0.05 to 0.55 and pick up 
    the lowest to control that. 
    '''
    N = affinity_matrix.shape[0]
    min_avg_degree = 2 * np.log(N)
    threshold = 0.05
    while threshold <= t:
        # threshold matrix
        affinity_threshold = threshold_proportional(affinity_matrix, threshold, copy=True)
        # convert to undirected graph
        G = nx.Graph()
        nodes = range(len(affinity_threshold))
        G.add_nodes_from(nodes)
        for i in range(affinity_threshold.shape[0]):
            for j in range(affinity_threshold.shape[1]):
                weight = affinity_threshold[i][j]  
                if weight != 0:
                    # G.add_edge(i, j, weight=weight)  
                    G.add_edge(i, j)
        avg_degree = sum(dict(G.degree()).values()) / len(G)
        print(threshold, avg_degree, min_avg_degree)
        if avg_degree > min_avg_degree:
            return threshold, G
        threshold += 0.05
    return None


# AVERAGE DEGREE
def av_degree(embedded_df, edges):
    G = nx.Graph()
    for idx, row in embedded_df.iterrows():
        G.add_node(idx, x=row['X'], y=row['Y'])
    G.add_edges_from(edges)
    # Calculate av_degree
    degrees = dict(G.degree())
    av_degree = sum(degrees.values()) / len(degrees)
    return av_degree


# PLOT NETWORK
def plot_network(embedded_df, edges):
    G = nx.Graph()
    for idx, row in embedded_df.iterrows():
        G.add_node(idx, pos=(row['X'], row['Y']))
    G.add_edges_from(edges)
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='skyblue')
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_color='black')
    # Display the graph
    plt.title('Graph Visualization')
    plt.show()
    
    
# PLOT NETWORK
def plot_networkx(G):
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=100, font_size=7, font_weight='bold', font_color='black', edge_color='gray', ax=ax)
    plt.show()




""" NETWORKS DYNAMIC THRESHOLD """

mes_network = {}
dist_semsim = {}
for key, mat in all_matrix_sim.items():
    t, G = select_threshold(mat, t=0.8)
    cc         = np.mean(list(nx.closeness_centrality(G).values()))
    density    = nx.density(G)
    ge         = nx.global_efficiency(G)
    aspl       = nx.average_shortest_path_length(G)
    clustering = nx.average_clustering(G)
    mean_sim   = mat_sim_measures(mat)[0]
    std_sim    = mat_sim_measures(mat)[1]
    mes_network[key] = [len(G), t, cc, density, ge, aspl, clustering, mean_sim, std_sim]
    dist_semsim[key] = mat_sim_dist(mat)

data_list = []
for key, value in mes_network.items():
    data_list.append({'name': key[:-7], 'group': get_group(key), 's_group': get_sgroup(key),
                      'task': key[-1],
                      'sem_sim': value[7],
                      'std_sim': value[8],
                      'n_nodes': value[0],
                      'threshold': value[1],
                      'c_centrality': value[2],
                      'density': value[3],
                      'efficiency': value[4],
                      'av_sh_pa_le': value[5],
                      'c_coeff': value[6]})

df_network = pd.DataFrame(data_list)
df_network['sem_sim'] = df_network['sem_sim'].astype(float)
df_network['std_sim'] = df_network['std_sim'].astype(float)
df_network.to_pickle(path_results+'df_n_all_thresholds.pkl')



""" DISTRIBUTION OF SEMSIM """

# FREQUENCIES OF ARRAY
def frequency(arr):
    freq = [0] * 21
    for value in arr:
        if 0 <= value <= 1:
            index = int(value * 20)  # Calculate the index corresponding to the value
            freq[index] += 1   # Increment the frequency count   
    return freq


# COMBINE ARRAYS
def combine_arrays(df, group, task):
    group_task_data = df[(df['group'] == group) & (df['task'] == task)]
    combined_array = []
    for array in group_task_data['dist_sem_sim']:
        combined_array.extend(array)
    return combined_array


# SEM SIMILARITIES ARRAYS
data_list = []
for key, value in  dist_semsim.items():
    data_list.append({'name': key[:-7], 'group': get_group(key), 's_group': get_sgroup(key),
                      'task': key[-1],
                      'dist_sem_sim': value})
df = pd.DataFrame(data_list)
unique_combinations  = df[['group', 'task']].drop_duplicates()
combined_arrays_dict = {}
for _, row in unique_combinations.iterrows():
    group, task = row['group'], row['task']
    combined_arrays_dict[(group, task)] = combine_arrays(df, group, task)    


# DICTIONARY OF FREQUENCIES
dict_freq = {}
for key, value in combined_arrays_dict.items():
    freq_dict = frequency(value)
    # Store the frequency dictionary in the new dictionary
    dict_freq[key] = freq_dict
del combined_arrays_dict




# PLOT DISTRIBUTION

exclude_keys = [('MD', '1'), ('MD', '2'), ('MD', '4'), ('MD', '6')]
def plot_arrays_by_number(data, task):
    x_values = np.arange(0, 1.05, 0.05)
    for key, value in data.items():
        if key[1] == task and key not in exclude_keys:
            plt.plot(x_values, value, label=key)
    plt.legend()
    plt.xlabel('Sem Sim')
    plt.ylabel('Frequency')
    plt.show()

plot_arrays_by_number(dict_freq, '1')
plot_arrays_by_number(dict_freq, '2')
plot_arrays_by_number(dict_freq, '4')
plot_arrays_by_number(dict_freq, '6')





"""
# NETWORK PROPERTIES USING THRESHOLD
m_network = {}
# OBTAIN CENTRALITY MEASURES    
for key, mat in all_matrix_sim.items():
    thres  = 0.75
    emb_df = get_umap(mat)
    edges  = generate_edges_t(mat, thres)
    m_network[key] = [centrality(emb_df, edges)[0], centrality(emb_df, edges)[1],
                      small_world(emb_df, edges),
                      mat_sim_measures(mat)[0], mat_sim_measures(mat)[1], mat_sim_measures(mat)[2]]
    
# NETWORK TO DATAFRAME
data_list = []
for key, value in m_network.items():
    centrality, clustering, sigma = value[0], value[1], value[2]
    mean, std, n_len = value[3], value[4], value[5]
    data_list.append({'name': key[:-7], 'group': get_group(key), 's_group': get_sgroup(key),
                      'task': key[-1],
                      'n_content': n_len, 'mean_sim': mean, 'std_sim': std,
                      'centrality': centrality, 'clustering': clustering, 'sigma': sigma})

df_network = pd.DataFrame(data_list)
df_network['mean_sim'] = df_network['mean_sim'].astype(float)
df_network['std_sim'] = df_network['std_sim'].astype(float)
df_network.to_pickle(path_results+'df_n_all.pkl')
"""


# AVERAGE VALUES (1,2,4,6)
non_numeric_df = df_network.select_dtypes(exclude=['number'])
non_numeric_df = non_numeric_df.drop_duplicates()
numeric_mean_df = df_network.groupby('name').mean().reset_index()
df_n_av = pd.merge(numeric_mean_df, non_numeric_df, on='name')
df_n_av = df_n_av[['name', 's_group', 'centrality', 'clustering', 'sigma']]
df_n_av = df_n_av.drop_duplicates().reset_index(drop=True)
df_n_av.to_pickle(path_results+'df_n_av.pkl')

# VALUES FOR EACH TASK
grouped = df_network.groupby('task')
grouped_dfs = {}
for name, group in grouped:
    grouped_dfs[name] = group.copy()

# FOR EACH TASK
df_n_1 = grouped_dfs['1'].reset_index(drop=True)
df_n_2 = grouped_dfs['2'].reset_index(drop=True)
df_n_4 = grouped_dfs['4'].reset_index(drop=True)
df_n_6 = grouped_dfs['6'].reset_index(drop=True)

# TO PICKLE
df_n_1.to_pickle(path_results+'df_n_1.pkl')
df_n_2.to_pickle(path_results+'df_n_2.pkl')
df_n_4.to_pickle(path_results+'df_n_4.pkl')
df_n_6.to_pickle(path_results+'df_n_6.pkl')



    
    
    
