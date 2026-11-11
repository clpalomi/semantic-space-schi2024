# Approximating the semantic space

Code supplied by Claudio Palominos for *Approximating the semantic space: Word embedding techniques in psychiatric speech analysis*, **Schizophrenia** (2024).

Publication: https://doi.org/10.1038/s41537-024-00524-7

The scripts are organized from metadata and speech measures to statistics and, finally, figures. They use the attached author scripts; no replacement analysis or clinical data has been added. This is a reconstruction of the supplied workflow, not a claim that the published numerical results have been independently reproduced.

## Before running

Use Python 3.10 or 3.11 for the supplied spaCy 3.6 model. Historical library versions and model revisions were not supplied. The dependency file lists inferred requirements, not the original environment lockfile.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.6.0/de_core_news_lg-3.6.0-py3-none-any.whl
mkdir -p data/texts data/metadata results/tables results/additional/umap results/additional/summaries
```

The original speech and clinical data cannot be distributed here. Prepare your own authorized data following [the input structure](docs/DATA_STRUCTURE.md). Files under `data/templates/` contain headers only; the scripts expect the Excel files described in the documentation.

All personal filesystem paths have been replaced with the generic paths in [pipeline/paths.py](pipeline/paths.py). Run commands **from the repository root**, using `python -m` as below. Input and generated output directories are ignored by Git.

The NLP stages require downloaded models: `aari1995/German_Semantic_STS_V2`, `dbmdz/bert-base-german-cased`, German fastText `cc.de.300.bin`, and spaCy `de_core_news_lg`. The first model downloads need network access and sufficient memory/disk for full embeddings. The existing fastText download calls use the current working directory. Do not substitute models between stages: the supplied alternatives produce different representations.

Stage 03 includes the author-approved tensor-to-NumPy conversion before the existing centroid calculation. Its pooling and centroid measures are tested with synthetic PyTorch outputs; full NLP model downloads and clinical results remain unverified.

## Analysis order

```bash
python -m pipeline.01_demographics
python -m pipeline.02_length
python -m pipeline.03_sentence_embeddings
python -m pipeline.04_word_embeddings
python -m pipeline.05_distance_norms
python -m pipeline.06_convex_hull
python -m pipeline.07_demographic_tables
python -m pipeline.08_semantic_regression
python -m pipeline.09_displacement_dispersion
python -m pipeline.10_group_differences
python -m pipeline.11_convex_hull_regression
```

Stage 01 creates `data/metadata/groups.xlsx`. If you already have an appropriately structured `groups.xlsx`, start at stage 02. Stage 05 contains additional distance-norm calculations; the subsequent main tables do not depend on it.

Stage 06 repeats the author's manually selected dimensionalities (3–10D) with the supplied German sentence model and reduction method. To calculate only the main 5D case:

```bash
python -m pipeline.06_convex_hull --dimensions 5
```

Run all dimensions before stage 11. Its regression summaries are printed to the terminal, as in the supplied code. To keep a report, redirect a stage's output to a text file under `results/`.

| Stage | Calculation | Main outputs under `results/` |
| --- | --- | --- |
| 02 | Speech and sentence length, group summaries, Bonferroni comparisons | `df_stats_length.pkl`, `summ_stats_length.xlsx`, `pvalue_stats_length.xlsx` |
| 03 | Sentence similarity, centroid, dispersion, displacement | `df_semsim_sent_length.pkl`, `df_centroids_sents_mono.pkl` |
| 04 | fastText and BERT similarity sequences and derived measures | `df_semsim_dist_length.pkl`, `df_centroids_words.pkl` |
| 05 | fastText distance norms | `df_distances_norms.pkl` |
| 06 | Sentence convex hulls by dimension | `df_sent_convexhull_{3..10}D_mono.pkl` |
| 07 | Demographic description and comparisons | `tables/table_1.xlsx`, `tables/table_2.xlsx` |
| 08 | Mixed regressions for semantic variables | `regression_table.xlsx`, `regression_table_signif.xlsx` |
| 09 | Word displacement and sentence dispersion/displacement regressions | Terminal summaries |
| 10 | Kruskal–Wallis group comparisons | `tables/table_4.xlsx` (Table S2 in the SM) |
| 11 | Log volume and log area regressions, 3–10D | Terminal summaries |

## Figures: final stage

After generating the required measures, run:

```bash
python -m pipeline.12_fig_centroids
python -m pipeline.13_fig_length_similarity
python -m pipeline.14_fig_group_effects
python -m pipeline.15_fig_symptoms
python -m pipeline.16_fig_convex_hull_illustration
python -m pipeline.17_fig_length
```

These scripts save figures under `results/`. For a machine without a display, prefix a command with `MPLBACKEND=Agg`. Stage 04 also retains the original per-sample slope diagnostic plots because its plotting function returns a feature used in the analysis.

| Figure script | Source identification / supplementary check |
| --- | --- |
| 12 | Figure 1: sentence-centroid t-SNE, picture/group separation; original in-sample KNN scores |
| 13 | Figure 2 and Figure S2: length versus mean semantic similarity |
| 14 | Figure 3: the author's fixed coefficients/p-values, matching Table S3; **not recomputed from a new dataset** |
| 15 | Figure 5 and Figure S3: symptom correlations; also the supplied partial-correlation plot |
| 16 | Convex-hull illustration using the supplied seeded random points; not clinical observations; figure number unconfirmed |
| 17 | Figure S1: length comparisons, using Bonferroni as specified in the SM |

Figure numbers in the main article follow the labels inside the supplied scripts. The Nature page could not be retrieved in this environment; the attached supplementary PDF was read and checked. See [reproducibility notes](docs/REPRODUCIBILITY.md) for source discrepancies and limits.

## Additional analyses and provenance

[additional/](additional/README.md) retains distinct earlier versions, perplexity/network analyses, multilingual comparisons, and participant-specific geometry plots. Their filenames retain the original ordering. They are not prerequisites for the main workflow and should not be run as a numbered batch. Their outputs are isolated under `results/additional/`.

Identical text helpers are shared in `pipeline/text_measures.py`. Numerical functions using script-specific model globals remain local. The repeated identical 9D calculation was removed. Calculation formulas, filters, model names, seeds and statistical conventions are otherwise preserved, apart from the explicitly approved corrections documented in [the change record](docs/REPRODUCIBILITY.md). [source_manifest.json](docs/source_manifest.json) records every supplied Python file's hash and the original line ranges used to organize the stages.

## Validation

```bash
python -m compileall -q pipeline additional
python -m unittest discover -s tests -v
```

The tests use temporary synthetic inputs to check text parsing, metadata/length stages, statistics exports and string-group symptom plotting. They do not download NLP models or reproduce clinical effect sizes. No original participant data or generated study results are committed.
