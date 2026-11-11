# Reproducibility and change record

## Evidence and scope

No participant data, original environment lockfile, model revision identifiers or original transcript-preprocessing script was supplied. The code starts from already prepared German transcripts. The header-only input templates describe what the scripts read; they do not recover the original data collection, exclusions or preprocessing.

## Main choices

- Sentence measures use the active German model `aari1995/German_Semantic_STS_V2` in `1_length_semsim_vars.py`. The SM describes 1024D sentence embeddings. The older multilingual 768D model remains an additional analysis and does not replace the German results.
- Word measures use the supplied fastText and `dbmdz/bert-base-german-cased` calculations, including their existing content-word filtering and tokenizer settings.
- The German convex-hull variant uses the active `reduce_dim_2`: pooled embeddings, standardization, Gaussian random projection, then t-SNE. Neither the inactive direct t-SNE nor PCA variants are substituted.
- Stage 06 repeats the supplied calculation for 3–10D, changing only the dimension, matching sentence-count thresholds, output filename and timing label. The source excludes `n_sent <= d` before reduction and assigns missing hull measures for `n_sent <= d+1`; this extra threshold is preserved. The 5D result feeds the main symptom/volume analyses. Stage 11 repeats the original log-volume/log-area regression for each dimension.
- Mixed models, predictors, random grouping, transformations and R-squared conventions stay as supplied. Different definitions of adjusted R-squared in different sections were not combined into one formula.

## Changes beyond moving files

1. Personal absolute paths were replaced by `pipeline/paths.py`. Generated pickle inputs are read from the results location of their producer instead of the original mixed data/results folders. Additional model variants write to `results/additional/` to avoid overwriting main results.
2. Main stages were separated at the existing section boundaries. Imports and loads of `df_stats_length.pkl` were added where needed to carry state between processes. Identical pure text helpers are shared without changing their bodies. Helpers relying on a script's active model remain local.
3. The duplicated 9D calculation in `2_convex_hull_german.py` was removed. The redundant manual pasted regression-output export is omitted from the main workflow. Repeated model calls from the table script are superseded by its own complete 12-variable regression/export section.
4. The hardcoded single-participant block in `1_length_semsim_vars.py` (original lines 657–710) was dropped from feature extraction. It referenced undefined `sem_list`/`sem_list_1` and interrupted the subsequent word loop. Final interactive diagnostics referring to earlier versions of `df` were not carried into the core producers.
5. Figure export calls already supplied but commented out in the length/similarity functions were enabled. Separate plot canvases prevent panels from overlaying; diagnostic filenames prevent overwriting the main length plots. No plotted values or regression calculations were changed.
6. Figure S1 uses Bonferroni instead of FDR-BH, matching the SM and the existing length-comparison table calculation.
7. With `group` stored as strings, symptom correlation matrices use numeric columns only. Partial-correlation p-values dummy-code group with `drop_first=True`, matching the supplied coefficient calculation. No ordinal group mapping was invented.


## Supplementary mapping

| Supplementary result | Reorganized source |
| --- | --- |
| Table S1 | Stage 02: group means/SD and Bonferroni comparisons |
| Figure S1 | Stage 17: length boxplots with the approved correction |
| Figure S2 | Stage 13: per-group BERT/fastText scatterplots |
| Table S2 | Stage 10: Kruskal–Wallis results; original filename remains `table_4.xlsx` |
| Table S3 | Stage 08: standardized coefficients/p-values; stage 14 visualizes the supplied fixed values |
| Table S4 | Stage 08: full mixed-model regression export |
| Table S5 | Stage 09: `sum_distances_n` regression |
| Table S6 | Stage 09: sentence `sum_dist` regression |
| Table S7 | Stage 11: 5D `log_area` regression |
| Table S8 | Stages 06 and 11: varying dimensions and log-volume regressions |
| Figure S3 | Stage 15: per-picture symptom correlation heatmaps |


## Resolved tensor compatibility correction

`sent_embedding_centroid` converts pooled embeddings with `.detach().cpu().numpy()` before the existing centroid, dispersion and distance formulas. This resolves the reproduced `TypeError: mean() got an unexpected keyword argument 'axis'` when NumPy receives a PyTorch tensor directly. A regression test exercises the actual pooling and centroid functions using synthetic PyTorch model outputs, verifies that padding is ignored, and checks centroid, cosine dispersion and cumulative Euclidean distance against independently specified expected values.

