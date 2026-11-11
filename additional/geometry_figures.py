from pipeline.text_measures import read_files_sent

# Participant-specific illustration; requires the original selected transcript.
from pipeline.paths import TEXTS, METADATA, ADDITIONAL_RESULTS

"""
@author: Claudio
"""

import os
import re
import numpy as np
import pandas as pd
import spacy
import scipy.ndimage
from scipy.spatial import Delaunay
import fasttext.util
from openTSNE import TSNE
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

# PATH INPUT
path_input = TEXTS
path_demo = METADATA
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




""" PROCESSING: SPLIT SPEECH INTO SENTENCES """

# SAVE EACH LIST OF WORDS
text_files = read_files_sent(path_input)
text_files = {key[:-4]: value for key, value in text_files.items()}
# IMPORT DEMOGRAPHICS
groups = pd.read_excel(path_demo+"groups.xlsx")
# UNIQUE NAMES
names = groups['name'].unique()
# Filter the 'text_files' dictionary based on the values in 'names'
text_files = {key: value for key, value in text_files.items() if key[:-7] in names}



""" SEMANTIC SIMILARITTY CONSECUTIVE WORDS """

# LOAD NLP SPACY
nlp = spacy.load("de_core_news_lg")

# LOAD FASTTEXT
fasttext.util.download_model('de', if_exists='ignore')  # German
ft = fasttext.load_model('cc.de.300.bin')


results = []
for key, lst in text_files.items():
    print(key)
    joined_list   = join_sentences(lst)
    c_words       = filter_content(joined_list)
    word_vectors  = [ft.get_word_vector(word) for word in c_words]
    points        = np.array(word_vectors)
    # REDUCTION
    tsne = TSNE(n_components=2, perplexity=45, n_jobs=-1, random_state=42)
    tsne_embeddings = tsne.fit(points)
    centroid_emb = [np.mean(list(tsne_embeddings[:, 0])), np.mean(list(tsne_embeddings[:, 1]))]
    # PLOT 1
    """
    plt.figure(figsize=(8, 6))
    #plt.set_palette("colorblind")
    plt.scatter(list(tsne_embeddings[:, 0]), list(tsne_embeddings[:, 1]), color='blue', s=10)
    #plt.scatter(centroid_emb[0], centroid_emb[1], color='red', marker='X', s=200)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('t-SNE word embeddings')
    plt.savefig(path_results+'words_'+key+'.png', dpi=500)
    plt.close()
    # PLOT 2
    plt.figure(figsize=(8, 6))
    #plt.set_palette("colorblind")
    plt.scatter(list(tsne_embeddings[:, 0]), list(tsne_embeddings[:, 1]), color='blue', s=10)
    plt.scatter(centroid_emb[0], centroid_emb[1], color='red', marker='X', s=200)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('t-SNE word embeddings')
    plt.savefig(path_results+'words_centroid_'+key+'.png', dpi=500)
    plt.close()
    """
    """ TO 3-D GRAVITY MODEL """
    x = np.array(list(tsne_embeddings[:, 0]))
    y = np.array(list(tsne_embeddings[:, 1]))
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    distances = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
    G = 1
    gravitational_potential = -G / (distances + 1e-9) 
    data_3d = np.vstack((x, y, gravitational_potential)).T
    # CURVATURE
    values = gravitational_potential
    grid_x, grid_y = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
    grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
    grid_z = scipy.interpolate.griddata((x, y), gravitational_potential, (grid_x, grid_y), method='cubic')
    # Apply Gaussian filter to smoothen the grid for derivative calculation
    dx, dy = np.gradient(grid_z)
    dxx, dxy = np.gradient(dx)
    dyx, dyy = np.gradient(dy)
    E = 1 + dx**2
    F = dx * dy
    G = 1 + dy**2
    L = dxx
    M = dxy
    N = dyy
    # Gaussian curvature
    K = ((L * N - M**2) / (E * G - F**2))
    total_gaussian_curvature = np.sum(K) * (grid_x[1,0] - grid_x[0,0]) * (grid_y[0,1] - grid_y[0,0])
    # Mean curvature
    H = ((E * N - 2 * F * M + G * L) / (2 * (E * G - F**2)))
    mean_curvature = np.mean(H)
    # PLOT
    # STATIC GRAPH
    """
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Gravitational Potential')
    ax.set_title('3D word embeddings with gravitational potential')
    plt.savefig(path_results+'words_3D_HC_'+key+'.png', dpi=500)
    ##  INTERACTIVE GRAPH
    trace_points = go.Scatter3d(
        x=data_3d[:, 0],
        y=data_3d[:, 1],
        z=data_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=data_3d[:, 2],
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Data Points'
    )
    layout = go.Layout(
        title='3D word embeddings with gravitational potential',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Gravitational Potential'),
        ),
        legend=dict(x=0.1, y=0.9)
    )
    fig = go.Figure(data=trace_points, layout=layout)
    pio.write_html(fig, file=path_results+'3d_scatter_words_'+key+'.html', auto_open=False)
    """
    # RESULTS
    results.append({'name': key, 'n_content': len(c_words),
                    'gaussian': total_gaussian_curvature, 'mean_curv': mean_curvature})
    

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
df.to_pickle(path_results+'df_curvature.pkl')





#lst=text_files['1341_2_bild_2']
#lst=text_files['0023_3_bild_2']

#
tsne = TSNE(n_components=2, perplexity=45, n_jobs=-1, random_state=42)
tsne_embeddings = tsne.fit(points)
tsne_embeddings = tsne_embeddings[(tsne_embeddings[:,0]>-5) & (tsne_embeddings[:,1]>-7.5)]
#tsne_embeddings = tsne_embeddings[(tsne_embeddings[:,0]<-5) & (tsne_embeddings[:,1]<-7.5)]
centroid_emb = [np.mean(list(tsne_embeddings[:, 0])), np.mean(list(tsne_embeddings[:, 1]))]

# PLOT
plt.figure(figsize=(8, 6))
#plt.set_palette("colorblind")
plt.scatter(list(tsne_embeddings[:, 0]), list(tsne_embeddings[:, 1]), color='blue', s=10)
#plt.scatter(centroid_emb[0], centroid_emb[1], color='red', marker='X', s=200)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('t-SNE word embeddings')
#plt.show()
plt.savefig(path_results+'words_without_centroid_1341_2.png', dpi=500)

plt.figure(figsize=(8, 6))
#plt.set_palette("colorblind")
plt.scatter(list(tsne_embeddings[:, 0]), list(tsne_embeddings[:, 1]), color='blue', s=10)
plt.scatter(centroid_emb[0], centroid_emb[1], color='red', marker='X', s=200)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('t-SNE word embeddings')
#plt.show()
plt.savefig(path_results+'words_with_centroid_1341_2.png', dpi=500)
#


""" TO 3-D GRAVITY MODEL """

x = np.array(list(tsne_embeddings[:, 0]))
y = np.array(list(tsne_embeddings[:, 1]))

centroid_x = np.mean(x)
centroid_y = np.mean(y)

distances = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)

G = 1
gravitational_potential = -G / (distances + 1e-9) 

data_3d = np.vstack((x, y, gravitational_potential)).T


# STATIC GRAPH
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c='b', marker='o')
#centroid_potential = -G / 1e-9  # very high value to represent infinite potential
#ax.scatter(centroid_x, centroid_y, centroid_potential, c='r', marker='x', s=100, label='Centroid')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Gravitational Potential')
ax.set_title('3D word embeddings with gravitational potential')
#plt.savefig(path_results+'words_3D_HC.png', dpi=500)
plt.show()


##  INTERACTIVE GRAPH
trace_points = go.Scatter3d(
    x=data_3d[:, 0],
    y=data_3d[:, 1],
    z=data_3d[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=data_3d[:, 2],
        colorscale='Viridis',
        opacity=0.8
    ),
    name='Data Points'
)

trace_lines = go.Scatter3d(
    x=data_3d[:, 0],
    y=data_3d[:, 1],
    z=data_3d[:, 2],
    mode='lines',
    line=dict(
        color='blue',
        width=2
    ),
    name='Consective crossing'
)

layout = go.Layout(
    title='3D word embeddings with gravitational potential',
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Gravitational Potential'),
    ),
    legend=dict(x=0.1, y=0.9)
)

#fig = go.Figure(data=trace_points, layout=layout)
fig = go.Figure(data=[trace_points, trace_lines], layout=layout)
pio.write_html(fig, file=path_results+'3d_scatter_plot_lines_1341_2.html', auto_open=True)


##  ANIMATED GRAPH
trace_points = go.Scatter3d(
    x=data_3d[:, 0],
    y=data_3d[:, 1],
    z=data_3d[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=data_3d[:, 2],
        colorscale='Viridis',
        opacity=0.8
    ),
    name='Data Points'
)

# Create initial trace for lines (empty initially)
trace_lines = go.Scatter3d(
    x=[],
    y=[],
    z=[],
    mode='lines',
    line=dict(
        color='blue',
        width=2
    ),
    name='Connecting Lines'
)

# Define frames for animation
frames = []
for i in range(1, len(x)):
    frame = go.Frame(
        data=[
            go.Scatter3d(
                x=data_3d[:i+1, 0],
                y=data_3d[:i+1, 1],
                z=data_3d[:i+1, 2],
                mode='lines',
                line=dict(
                    color='blue',
                    width=2,
                    dash='dash'
                )
            )
        ],
        name=f'frame{i}'
    )
    frames.append(frame)

# Create layout with animation settings
layout = go.Layout(
    title='3D Scatter Plot with Sequential Connecting Lines Animation',
    scene=dict(
        xaxis=dict(title='X Axis'),
        yaxis=dict(title='Y Axis'),
        zaxis=dict(title='Gravitational Potential'),
    ),
    updatemenus=[{
        'type': 'buttons',
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
        }, {
            'label': 'Pause',
            'method': 'animate',
            'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
        }]
    }]
)

# Combine data and layout into a figure
fig = go.Figure(data=[trace_points, trace_lines], layout=layout, frames=frames)

# Save the plot as an HTML file
pio.write_html(fig, file='3d_scatter_with_animated_lines.html', auto_open=True)



## ANIMATED GRAPH 2
trace_points = go.Scatter3d(
    x=data_3d[:, 0],
    y=data_3d[:, 1],
    z=data_3d[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=data_3d[:, 2],
        colorscale='Viridis',
        opacity=0.8
    ),
    name='Data Points'
)

# Create initial trace for lines (empty initially)
trace_lines = go.Scatter3d(
    x=[],
    y=[],
    z=[],
    mode='lines',
    line=dict(
        color='blue',
        width=3,
        dash='dashdot'  # dash style: solid, dot, dash, longdash, dashdot, longdashdot
    ),
    name='Connecting Lines'
)

# Define frames for animation
frames = []
for i in range(1, len(x)):
    frame = go.Frame(
        data=[
            go.Scatter3d(
                x=[data_3d[i-1, 0], data_3d[i, 0]],
                y=[data_3d[i-1, 1], data_3d[i, 1]],
                z=[data_3d[i-1, 2], data_3d[i, 2]],
                mode='lines',
                line=dict(
                    color='blue',
                    width=3,
                    dash='dashdot'  # dash style: solid, dot, dash, longdash, dashdot, longdashdot
                )
            )
        ],
        name=f'frame{i}'
    )
    frames.append(frame)

# Create layout with animation settings
layout = go.Layout(
    title='3D Scatter Plot with Animated Connecting Lines',
    scene=dict(
        xaxis=dict(title='X Axis'),
        yaxis=dict(title='Y Axis'),
        zaxis=dict(title='Gravitational Potential'),
    ),
    updatemenus=[{
        'type': 'buttons',
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
        }, {
            'label': 'Pause',
            'method': 'animate',
            'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
        }]
    }]
)

# Combine data and layout into a figure
fig = go.Figure(data=[trace_points, trace_lines], layout=layout, frames=frames)

# Save the plot as an HTML file
pio.write_html(fig, file='3d_scatter_with_animated_lines.html', auto_open=True)



## SURFACE
from scipy.interpolate import griddata

points = np.vstack((x, y)).T
values = gravitational_potential
grid_x, grid_y = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

surface = go.Surface(
    x=grid_x,
    y=grid_y,
    z=grid_z,
    colorscale='Viridis'
)
layout = go.Layout(
    title='3D word embeddings with gravitational potential',
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Gravitational Potential'),
    )
)
fig = go.Figure(data=[surface], layout=layout)
pio.write_html(fig, file=path_results+'3d_surface_plot_HC.html', auto_open=True)
