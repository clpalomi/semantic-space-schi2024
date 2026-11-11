# This figure uses the fixed coefficients supplied by the author; it does not refit models.
from pipeline.paths import METADATA, RESULTS, TABLES

"""
@author: Claudio

Tables and results used in the article

"""

import matplotlib.pyplot as plt
import numpy as np

# PATH
path_demo = METADATA
path_tables = TABLES
path_results = RESULTS

variables = ['mean semsim', 'max semsim', 'min semsim', 'ssc', 'mean crossing rate', 'autocorrelation']
models = ['FastText MDD', 'FastText SSD', 'BERT MDD', 'BERT SSD']
coefficients = [
    [0.09, 0.03, -0.15, -0.09],
    [-0.18, -0.08, 0.07, 0.28],
    [0.09, -0.08, -0.10, -0.11],
    [-0.10, -0.08, -0.03, -0.37],
    [-0.33, -0.14, -0.15, -0.23],
    [0.29, 0.04, 0.17, 0.24]
]
p_values = [
    [0.541, 0.835, 0.258, 0.520],
    [0.105, 0.478, 0.549, 0.018],
    [0.392, 0.464, 0.383, 0.336],
    [0.417, 0.493, 0.856, 0.023],
    [0.005, 0.241, 0.240, 0.087],
    [0.016, 0.733, 0.224, 0.084]
]

fig, ax = plt.subplots(figsize=(12, 8))
width = 0.2
x = np.arange(len(variables))
colors = {'FastText': 'blue', 'BERT': 'green'}
shades = [0.4, 0.9]  # different shades of the same color
for i, model in enumerate(models):
    group = 'FastText' if 'FastText' in model else 'BERT'
    shade = shades[0] if 'MDD' in model else shades[1]
    color = colors[group]
    coeff = [row[i] for row in coefficients]
    bars = ax.bar(x + i * width, coeff, width, label=model, color=color, alpha=shade)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(variables, rotation=45, ha='right', fontsize=14)
ax.set_ylabel('Standardized coefficients', fontsize=14)
#ax.set_title('Standardized Coefficients for different models')
ax.legend(fontsize=12)

# Annotate significant p-values
significance_threshold = 0.05
for i, model in enumerate(models):
    for j, p_val in enumerate(p_values):
        if p_val[i] < significance_threshold:
            y = coefficients[j][i]
            offset = 0.005 if y > 0 else -0.01
            ax.text(x[j] + i * width, y + offset, f'p={p_val[i]:.3f}', ha='center', va='bottom' if y > 0 else 'top')
fig.tight_layout()
plt.savefig(path_results+'summary_group_effects_semsim.jpg', dpi=500)
#plt.show()

