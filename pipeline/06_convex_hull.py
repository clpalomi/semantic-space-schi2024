import argparse

parser = argparse.ArgumentParser(description="Sentence convex hulls for each selected dimension.")
parser.add_argument("--dimensions", nargs="+", type=int, choices=range(3, 11), default=list(range(3, 11)))
args = parser.parse_args()

from pipeline.text_measures import read_files_sent

# Repeat the supplied calculation for the dimensions selected by the author.
from pipeline.paths import TEXTS, METADATA, RESULTS, SANTANDER

"""
@author: Claudio
"""

import numpy as np
import pandas as pd
import time
import torch
from openTSNE import TSNE
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
from scipy.spatial import ConvexHull


# PATHS
path_input = TEXTS
path_demo = METADATA
path_santander = SANTANDER
path_results = RESULTS


""" FUNCTIONS """


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


from sklearn.random_projection import GaussianRandomProjection
# REDUCE DIMENSIONALITY SENTENCES (changed for use with torch)
def reduce_dim_2(sentences, dim=5):
    # Tokenize sentences
    encoded_input = sbert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    #sentence_embeddings = model.encode(sentences)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    input_dim = sentence_embeddings.shape[1]
    reduced_dim = input_dim // 4  # Reduce to half
    # Scale the embeddings
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(sentence_embeddings)
    grp = GaussianRandomProjection(n_components=reduced_dim, random_state=42)
    reduced_embeddings = grp.fit_transform(scaled_embeddings)
    tsne = TSNE(n_components=dim, perplexity=4, n_jobs=-1, random_state=42)
    tsne_embeddings = tsne.fit(reduced_embeddings)
    tsne_embeddings = np.array(tsne_embeddings)
    return tsne_embeddings


# NORMALIZE EMBEDDINGS
def normalize_embeddings(embeddings):
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    return scaled_embeddings


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

# model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')  # old model
# model = SentenceTransformer('aari1995/German_Semantic_STS_V2')        # alternative model

sbert_tokenizer = AutoTokenizer.from_pretrained('aari1995/German_Semantic_STS_V2')
model           = AutoModel.from_pretrained('aari1995/German_Semantic_STS_V2')


for dimension in args.dimensions:
    # START-RUN
    start_time = time.time()

    # OBTAIN REDUCED POINTS
    results = []
    for key, lst in text_german.items():
        print(key)
        #lst = remove_short_sentences(lst)
        n_sent  = len(lst)
        if n_sent <= dimension:
            points          = np.nan
            #centroid       = np.nan
            #dispersion     = np.nan
            #sum_distances  = np.nan
        else:
            points          = reduce_dim_2(lst, dimension)
            #centroid       = sent_embedding_centroid(lst)[0]
            #dispersion     = sent_embedding_centroid(lst)[1]
            #sum_distances  = sent_embedding_centroid(lst)[2]
        results.append({'name':  key, 'n_sent': n_sent,
                        'points': points
                        })

    results = [item for item in results if isinstance(item.get('points'), np.ndarray) and not np.any(np.isnan(item.get('points')))]

    # ALL POINTS
    all_points = np.vstack([item['points'] for item in results])

    # NORMALIZE ALL POINTS
    normalized_all_points = normalize_embeddings(all_points)

    # DISTRIBUTE NORMALIZED POINTS BACK
    start = 0
    for item in results:
        n_sent = item['n_sent']
        end = start + n_sent
        item['normalized_points'] = normalized_all_points[start:end]
        start = end

    # ALL RESULTS
    results_all = []
    for item in results:
        print(item['name'])
        normalized_points = item['normalized_points']
        points = item['points']
        n_sent = len(normalized_points)
        if n_sent <= dimension + 1:
            area     = np.nan
            volume   = np.nan
            area_n   = np.nan
            volume_n = np.nan
        else:
            convex_hull = ConvexHull(points)
            area = convex_hull.area
            volume = convex_hull.volume
            convex_hull_n = ConvexHull(normalized_points)
            area_n = convex_hull_n.area
            volume_n = convex_hull_n.volume
    
        results_all.append({
            'name': item['name'],
            'n_sent': n_sent,
            'area': area,
            'volume': volume,
            'area_n': area_n,
            'volume_n': volume_n
        })

    # RESULTS DATAFRAME
    df = pd.DataFrame(results_all)
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
    df.to_pickle(path_results+f'df_sent_convexhull_{dimension}D_mono.pkl')
    # df = pd.read_pickle(path_results+'df_sent_convexhull.pkl')

    # END-RUN
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print(f"{dimension}-D Calculations took {elapsed_minutes:.2f} minutes to execute.")

