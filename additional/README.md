# Additional analyses supplied by the author

These scripts preserve distinct analyses and earlier variants. They are outside the main replication sequence because the supplied final table script and SM identify a different primary workflow, or because their upstream inputs are absent. Do not run this directory as a numbered batch.

Run modules from the repository root, e.g. `python -m additional.1_semsim_sz`. Install `requirements-additional.txt` if using the optional network or Plotly analyses. Generic paths are shared with `pipeline/paths.py`; variant outputs go to `results/additional/`, network outputs to its `umap/` subfolder. Keep variants' generated files separate from the main results.

| Supplied file | Purpose and dependencies |
| --- | --- |
| `1_semsim_sz.py` | Earlier BERT semantic-similarity summaries; prepared German texts and groups |
| `2_perplexity_sz.py` | German GPT-2 perplexity/token-type measures; texts, groups, spaCy, `dbmdz/german-gpt2` |
| `3_matrix_semsim_sz.py` | BERT similarity matrices, UMAP and graph measures; texts/groups, UMAP, NetworkX |
| `4_aggregated_relations.py` | Earlier perplexity/similarity/symptom merges; **`df_semsim_german.pkl` has no supplied producer** |
| `5_summary_differences.py` | CLI group-comparison summaries of earlier pickles |
| `5_summary_differences_mixed_effects.py` | Earlier mixed-model CLI; includes interactive example code before argument parsing |
| `6_run_several_summ.py` | Runs the pairwise summary batches for the earlier files; run only after their producers |
| `7_append_summaries.py` | Combines generated summary workbooks |
| `7_symptoms.py` | Earlier BERT/symptom analysis, including `Sz` label assumptions |
| `8_length.py` | Earlier BERT-only length/semantic features; superseded by the more complete main producer |
| `8_length_modified.py` | Earlier multilingual sentence-model variant plus word measures; writes 768D sentence centroids |
| `9_perp_length_semsim.py` | Perplexity/length/semantic relationships; requires the optional perplexity and feature outputs |
| `11_tSNE_centroids.py` | Word-centroid t-SNE displays; requires the optional word-centroid pickle |
| `12_analysis_all_var.py` | Recurrence/semantic relationships; **`df_recurrences.pkl` has no supplied producer** |
| `31_centroids_santander.py` | Spanish/German multilingual sentence comparison; requires Santander workbook and older German 768D centroids |
| `32_convex_hull_german.py` | Earlier multilingual 5D hulls; exploratory group tests |
| `2_convex_hull_german.py` | Multilingual 9D/10D hull variant; duplicate 9D block removed |
| `61_diff_coref_models.py` | Comparison of precomputed `CONLL_AV_TuBA`/`CONLL_AV_C2F_DROC` columns; their score-generation code is not supplied |
| `geometry_figures.py` | Original `71_geometry_centroids.py`: selected-transcript fastText geometry/curvature plots; hardcoded participant selections and interactive state |
| `semsim_symptoms.py` | Earlier sentence/symptom plots; older missing-input and dataframe-selection assumptions |
| `tables_other_models.py` | Extra sections from `51_tables_results.py`: interactive model variations and corrected-p-value experiments; refers to unresolved `df_sents`/`area_n` state and an older 5D filename |

The optional Santander workbook requires `PAR`, `Picture`, `Group`, and `utterance`. The source maps picture values `picture_1`/`picture_2`/`picture_3` to `1`/`2`/`3`, and group values `controles`/`pacientes` to `HC`/`SZ`. Its German comparison expects the multilingual `df_centroids_sents.pkl`, not the main 1024D monolingual centroid file.

Original string group labels differ between variants (`Sz`, `MD`, `SZ` versus `SSD`, `MDD`). Original pandas calls such as `groupby(...).mean()` over mixed string/numeric columns can require older pandas behavior. These are preserved, not implicitly harmonized. Several convex-hull diagnostic blocks use undefined `df_1`, `df_2`, `df_4`, `df_6` from an interactive session after saving their results. They do not form a fully automated alternative pipeline.

Only main symptom stages received the explicitly approved string-group correction. Unverified exploratory variants remain visible for provenance rather than being advertised as working replicas. The original preprocessing, coreference-score generation and recurrence extraction scripts would be needed to complete those branches; their absence does not create a dependency for the main semantic-space workflow.
