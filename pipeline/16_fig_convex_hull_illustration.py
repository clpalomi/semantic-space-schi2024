# Illustration with the random points in the supplied script; these are not study data.
from pipeline.paths import RESULTS

# -*- coding: utf-8 -*-
"""
@author: Claudio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


# PATHS
path_results = RESULTS

# RANDOM POINTS
np.random.seed(42)  # for reproducibility
points = np.random.uniform(0, 20, size=(25, 2))

# DATAFRAME WITH DATA
df = pd.DataFrame(points, columns=['X', 'Y'])
df['Group'] = np.repeat(np.arange(1, 6), 5)


""" FUNCTIONS """

def plot_point(df, group, point_index, title):
    plt.figure()
    group_data = df[(df['Group'] == group) & (df.index % 5 == point_index)]
    lower_index_points = df[(df['Group'] == group) & (df.index % 5 < point_index)]
    plt.scatter(lower_index_points['X'], lower_index_points['Y'], color='blue', alpha=0.5)
    plt.scatter(group_data['X'], group_data['Y'], color='blue')
    x_min, x_max = (0, 20)
    y_min, y_max = (0, 20)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(path_results+f'sent_emb_red_{title}_{group}.jpg', dpi=500)
    plt.close()
    # plt.show()

def plot_group_points(df, group):
    plt.figure()
    group_data = df[df['Group'] == group]
    plt.scatter(group_data['X'], group_data['Y'])
    x_min, x_max = (0, 20)
    y_min, y_max = (0, 20)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Sentences in a lower-dimensional space')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(path_results+f'sent_emb_red_{group}.jpg', dpi=500)
    plt.close()
   #  plt.show()

def plot_convex_hull(df, group):
    plt.figure()
    group_data = df[df['Group'] == group]
    hull = ConvexHull(group_data[['X', 'Y']])
    lower_index_points = df[(df['Group'] == group) & (df.index % 5 < 5)]
    plt.scatter(lower_index_points['X'], lower_index_points['Y'], color='blue', alpha=0.5)
    plt.scatter(group_data['X'], group_data['Y'])
    x_min, x_max = (0, 20)
    y_min, y_max = (0, 20)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    for simplex in hull.simplices:
        plt.plot(group_data['X'].iloc[simplex], group_data['Y'].iloc[simplex], 'k-')
    plt.title('Convex Hull of sentences')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(path_results+f'sent_emb_hull_{group}.jpg', dpi=500)
    plt.close()
    # plt.show()

def plot_centroid(df, group):
    plt.figure()
    group_data = df[df['Group'] == group]
    centroid = group_data[['X', 'Y']].mean()
    lower_index_points = df[(df['Group'] == group) & (df.index % 5 < 5)]
    plt.scatter(lower_index_points['X'], lower_index_points['Y'], color='blue', alpha=0.5)
    plt.scatter(group_data['X'], group_data['Y'], label=f'Group {group}')
    plt.scatter(centroid['X'], centroid['Y'], color='red', marker='X', s=100)
    x_min, x_max = (0, 20)
    y_min, y_max = (0, 20)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Centroid of sentences in a lower-dimensional space')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(path_results+f'sent_emb_centroid_{group}.jpg', dpi=500)
    plt.close()
    #plt.show()

def plot_selected_groups(df, selected_groups):
    fig, ax = plt.subplots(figsize=(8, 8))
    for group in selected_groups:
        group_data = df[df['Group'] == group]
        ax.scatter(group_data['X'], group_data['Y'], label=f'Speech {group}')
        hull = ConvexHull(group_data[['X', 'Y']])
        for simplex in hull.simplices:
            ax.plot(group_data['X'].iloc[simplex], group_data['Y'].iloc[simplex], 'k-')
        centroid = group_data[['X', 'Y']].mean()
        ax.scatter(centroid['X'], centroid['Y'], color='red', marker='X', s=100, label=f'Centroid {group}')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_title('Different speeches and centroids')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.savefig(path_results+f'sent_emb_red_groups_{selected_groups}.jpg', dpi=500)
    plt.close()
    #plt.show()


""" PLOTS """

plot_point(df, 1, 0, 'Sentence 1')
plot_point(df, 1, 1, 'Sentence 2')
plot_point(df, 1, 2, 'Sentence 3')
plot_point(df, 1, 3, 'Sentence 4')
plot_point(df, 1, 4, 'Sentence 5')
plot_convex_hull(df, 1)

plot_point(df, 2, 0, 'Sentence 1')
plot_point(df, 2, 1, 'Sentence 2')
plot_point(df, 2, 2, 'Sentence 3')
plot_point(df, 2, 3, 'Sentence 4')
plot_point(df, 2, 4, 'Sentence 5')
plot_convex_hull(df, 2)

plot_group_points(df, 1)
plot_centroid(df, 1)

plot_group_points(df, 2)
plot_centroid(df, 2)


# GROUPS TOGETHER

selected_groups = [1]
plot_selected_groups(df, selected_groups)
selected_groups = [2]
plot_selected_groups(df, selected_groups)
selected_groups = [1, 2]
plot_selected_groups(df, selected_groups)
selected_groups = [1, 2, 3]
plot_selected_groups(df, selected_groups)
selected_groups = [1, 2, 3, 4]
plot_selected_groups(df, selected_groups)
selected_groups = [1, 2, 3, 4, 5]
plot_selected_groups(df, selected_groups)

