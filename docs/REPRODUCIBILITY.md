# Reproducibility and change record

## Evidence and scope

The repository was organized using the 26 supplied Python files. The supplied supplementary material, *Approximating the semantic space – Supplementary material.pdf*, was read to check the supplementary tables and figures. Access to the Nature article was blocked; main-article numbering follows the labels in `51_tables_results.py`, rather than an independent inspection of the article.

No participant data, original environment lockfile, model revision identifiers or original transcript-preprocessing script was supplied. The code starts from already prepared German transcripts. The header-only input templates describe what the scripts read; they do not recover the original data collection, exclusions or preprocessing.

## Main choices

- Sentence measures use the active German model `aari1995/German_Semantic_STS_V2` in `1_length_semsim_vars.py`. The SM describes 1024D sentence embeddings. The older multilingual 768D model remains an additional analysis and does not replace the German results.
- Word measures use the supplied fastText and `dbmdz/bert-base-german-cased` calculations, including their existing content-word filtering and tokenizer settings.
- The German convex-hull variant uses the active `reduce_dim_2`: pooled embeddings, standardization, Gaussian random projection, then t-SNE. Neither the inactive direct t-SNE nor PCA variants are substituted.
- The author confirmed that dimensions were changed manually. Stage 06 repeats the supplied calculation for 3–10D, changing only the dimension, matching sentence-count thresholds, output filename and timing label. The source excludes `n_sent <= d` before reduction and assigns missing hull measures for `n_sent <= d+1`; this extra threshold is preserved. The 5D result feeds the main symptom/volume analyses. Stage 11 repeats the original log-volume/log-area regression for each dimension.
- Mixed models, predictors, random grouping, transformations and R-squared conventions stay as supplied. Different definitions of adjusted R-squared in different sections were not combined into one formula.

## Changes beyond moving files

1. Personal absolute paths were replaced by `pipeline/paths.py`. Generated pickle inputs are read from the results location of their producer instead of the original mixed data/results folders. Additional model variants write to `results/additional/` to avoid overwriting main results.
2. Main stages were separated at the existing section boundaries. Imports and loads of `df_stats_length.pkl` were added where needed to carry state between processes. Identical pure text helpers are shared without changing their bodies. Helpers relying on a script's active model remain local.
3. The duplicated 9D calculation in `2_convex_hull_german.py` was removed. The redundant manual pasted regression-output export is omitted from the main workflow. Repeated model calls from the table script are superseded by its own complete 12-variable regression/export section.
4. The hardcoded single-participant block in `1_length_semsim_vars.py` (original lines 657–710) was dropped from feature extraction. It referenced undefined `sem_list`/`sem_list_1` and interrupted the subsequent word loop. Final interactive diagnostics referring to earlier versions of `df` were not carried into the core producers.
5. Figure export calls already supplied but commented out in the length/similarity functions were enabled. Separate plot canvases prevent panels from overlaying; diagnostic filenames prevent overwriting the main length plots. No plotted values or regression calculations were changed.
6. **Author-approved:** Figure S1 now uses Bonferroni instead of FDR-BH, matching the SM and the existing length-comparison table calculation.
7. **Author-approved:** With `group` stored as strings, symptom correlation matrices use numeric columns only. Partial-correlation p-values dummy-code group with `drop_first=True`, matching the supplied coefficient calculation. No ordinal group mapping was invented.

The provenance manifest records original attachment hashes and source ranges. Some source ranges overlap because independent stages need their original imports or preparation blocks; identical pure helper implementations occur once in the main pipeline.

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

The supplied script calls the main volume/area regressions on `log_volume`/`log_area`. The SM's Table S7 caption says area without making that transformation explicit; the code's transformation is retained. No reported coefficients were substituted into refitted regressions. The Figure 3 constants, already present in the author script, match the SM's Table S3 and are clearly labeled as fixed values.

## Resolved tensor compatibility correction

**Author-approved and applied:** `sent_embedding_centroid` converts pooled embeddings with `.detach().cpu().numpy()` before the existing centroid, dispersion and distance formulas. This resolves the reproduced `TypeError: mean() got an unexpected keyword argument 'axis'` when NumPy receives a PyTorch tensor directly. A regression test exercises the actual pooling and centroid functions using synthetic PyTorch model outputs, verifies that padding is ignored, and checks centroid, cosine dispersion and cumulative Euclidean distance against independently specified expected values. The language model itself is not downloaded by this test.

## Remaining source-level limitations

- Exact published values require the unavailable original prepared transcripts, clinical data, participant eligibility and historical model/software versions.
- Some short-speech branches in the supplied feature code are incomplete. For example, the sentence loop's one-sentence branch does not assign every difference statistic later appended; `crossings_average` divides by `len(sequence)-1`. These edge cases are preserved rather than silently introducing new missing-data or exclusion rules.
- `centroid_fasttext` starts from index 1; the reported `kurt` subtracts 3 after `scipy.stats.kurtosis`; BERT tokenization applies truncation before the later chunk loop. These original conventions are retained and may merit separate methodological review. They were not replaced with different definitions.
- The supplied t-SNE/KNN separability code fits and predicts on the same observations. Its printed scores are in-sample descriptions, not held-out validation.
- Partial-correlation p-values retain the source's Pearson test on residuals. No new multiple-testing or degrees-of-freedom procedure was introduced. Heatmaps display significant cells and encode other cells as zero, following the original code.
- Several exploratory scripts require unavailable recurrence/coreference intermediates or interactive variables. They are listed in `additional/README.md` and are not included in the main run sequence.

## Validation evidence

All reorganized Python files were checked for syntax. Main-stage imports/name references were checked with Pyflakes. Synthetic smoke checks exercise input parsing, metadata and length outputs, mixed-regression table exports, dimension-specific regression inputs and symptom plots with string groups. Independent fixed-coefficient and seeded-point figures can be generated without research data.

These checks establish code organization and executable data-independent paths. They do not validate clinical effect sizes, statistical convergence on the original dataset or end-to-end NLP model downloads. No successful published-result replication is claimed.

The reduction helper was also exercised with actual PyTorch tensors, synthetic embeddings and openTSNE 1.0.4 at 3D, 5D, 8D and 10D; all produced finite positive convex-hull volumes. openTSNE emitted its warning that Barnes–Hut above 3D may be removed in a future version. The dependency is therefore pinned to the checked 1.0.4 version, without changing the author's reduction method. Fixed-coefficient and seeded-hull figure scripts generated 23 image files in a temporary directory. These checks used Python 3.12 and current numeric libraries; the full legacy spaCy/NLP environment was not installed or validated.
