"""Data-independent checks of the reorganized stages; no clinical data or models."""

from pathlib import Path
import runpy
import io
from contextlib import redirect_stdout
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline import paths
from pipeline.text_measures import read_files_sent, sentence_lengths, length_statistics


class PipelineSmokeTest(unittest.TestCase):
    def test_demographics_to_length_and_tables(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            texts, metadata, results, tables = [root / n for n in ('texts', 'metadata', 'results', 'tables')]
            for directory in (texts, metadata, results, tables):
                directory.mkdir()
            rows = []
            for g, group in enumerate(('HC', 'MDD', 'SSD')):
                for i in range(5):
                    name = f'P{g}{i}_session'
                    rows.append(dict(Proband=name, Typ2=group, Typ=group,
                                     SKID_Diagnose_1=group, Alter=25+i+g,
                                     Geschlecht=1+i % 2, Bildungsjahre=10+i,
                                     IQ=95+2*i+g, AgeOfOnset=20+i))
                    for picture in (1, 2, 4, 6):
                        speech = ' '.join(['Ein Mensch geht.'] * (3+i+g+picture))
                        (texts / f'{name}_bild_{picture}.txt').write_text(speech, encoding='utf-8')
            pd.DataFrame(rows).to_csv(metadata / 'ssd_mdd_hc_data.csv', index=False)
            bindings = dict(TEXTS=str(texts)+'/', METADATA=str(metadata)+'/',
                            RESULTS=str(results)+'/', TABLES=str(tables)+'/')
            with patch.multiple(paths, **bindings):
                for module in ('01_demographics', '02_length', '07_demographic_tables'):
                    with redirect_stdout(io.StringIO()):
                        runpy.run_module('pipeline.'+module, run_name='__main__')
            groups = pd.read_excel(metadata / 'groups.xlsx')
            self.assertEqual(len(groups), 15)
            stats = pd.read_pickle(results / 'df_stats_length.pkl')
            self.assertEqual(len(stats), 60)
            self.assertEqual(set(stats['group']), {'HC', 'MDD', 'SSD'})
            self.assertTrue(stats['total_length'].gt(0).all())
            self.assertTrue(np.allclose(stats['sent_av_length'], 3))
            pvalues = pd.read_excel(results / 'pvalue_stats_length.xlsx')
            self.assertEqual(len(pvalues), 24)
            self.assertEqual(set(pvalues['bild']), {1, 2, 4, 6})
            self.assertTrue((tables / 'table_1.xlsx').is_file())
            self.assertTrue((tables / 'table_2.xlsx').is_file())

    def test_transcript_reader_and_lengths(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p / 'P001_session_bild_1.txt').write_text('Ein Mensch geht\nEr bleibt hier', encoding='utf-8')
            (p / 'unrelated.txt').write_text('Ignored.', encoding='utf-8')
            speech = read_files_sent(folder)
            self.assertEqual(list(speech), ['P001_session_bild_1.txt'])
            stats = length_statistics(sentence_lengths(speech))
            self.assertEqual(stats.loc[0, 'total_length'], 6)
            self.assertEqual(stats.loc[0, 'sent_av_length'], 3)


class StatisticsSmokeTest(unittest.TestCase):
    def test_regression_exports_and_string_group_symptoms(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from itertools import product
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            metadata = root / 'metadata'
            metadata.mkdir()
            rng = np.random.default_rng(42)
            rows = []
            for group, i, picture in product(('HC', 'MDD', 'SSD'), range(10), ('1', '2', '4', '6')):
                rows.append(dict(name=f'{group}_{i}', group=group, bild=picture))
            words = pd.DataFrame(rows)
            for col in ('n_content', 'total_length', 'sent_av_length', 'sum_distances', 'sum_distances_n',
                        'mean_semsim', 'max_semsim', 'min_semsim', 'av_ssc', 'av_crossing', 'autocorr', 'skew',
                        'mean_semsim_t', 'max_semsim_t', 'min_semsim_t', 'av_ssc_t', 'av_crossing_t', 'autocorr_t', 'skew_t'):
                effects = dict(zip(words['name'].unique(), rng.normal(0, 1, words['name'].nunique())))
                words[col] = 8 + words['name'].map(effects) + rng.normal(0, .5, len(words))
            words.to_pickle(root / 'df_semsim_dist_length.pkl')
            sents = words[['name', 'group', 'bild', 'total_length', 'sent_av_length']].copy()
            for col in ('n_sent', 'dispersion', 'sum_dist'):
                effects = dict(zip(sents['name'].unique(), rng.normal(0, 1, sents['name'].nunique())))
                sents[col] = 8 + sents['name'].map(effects) + rng.normal(0, .5, len(sents))
            sents.to_pickle(root / 'df_semsim_sent_length.pkl')
            hull = sents[['name', 'group', 'bild', 'n_sent']].copy()
            hull['volume'] = rng.uniform(1, 10, len(hull))
            hull['area'] = rng.uniform(1, 10, len(hull))
            for dim in range(3, 11):
                hull.to_pickle(root / f'df_sent_convexhull_{dim}D_mono.pkl')
            header = (Path(__file__).resolve().parents[1] / 'data/templates/symptoms.csv').read_text().strip().split(',')
            symptoms = pd.DataFrame({'name': words['name'].unique()})
            for col in header[1:]:
                symptoms[col] = rng.uniform(1, 10, len(symptoms))
            symptoms.to_excel(metadata / 'symptoms.xlsx', index=False)
            bindings = dict(RESULTS=str(root)+'/', TABLES=str(root)+'/', METADATA=str(metadata)+'/')
            # Suppress only report text; errors and model warnings remain visible.
            with patch.multiple(paths, **bindings), redirect_stdout(io.StringIO()):
                for module in ('08_semantic_regression', '09_displacement_dispersion', '10_group_differences', '11_convex_hull_regression'):
                    runpy.run_module('pipeline.'+module, run_name='__main__')
                # Exercise the real string-group correction and plots with low-resolution temporary output.
                savefig = plt.savefig
                def small_savefig(*args, **kwargs):
                    kwargs['dpi'] = 35
                    return savefig(*args, **kwargs)
                with patch.object(plt, 'savefig', side_effect=small_savefig):
                    namespace = runpy.run_module('pipeline.15_fig_symptoms', run_name='__main__')
            self.assertEqual(len(pd.read_excel(root / 'regression_table.xlsx')), 108)
            self.assertEqual(len(pd.read_excel(root / 'regression_table_signif.xlsx')), 24)
            self.assertEqual(len(pd.read_excel(root / 'table_4.xlsx')), 12)
            self.assertTrue(np.isfinite(namespace['partial_corr_matrix'].loc['mean_semsim', 'log_volume']))
            self.assertEqual(len(list(root.glob('heatmap*.png'))), 6)
            plt.close('all')


class SentenceCentroidTest(unittest.TestCase):
    def test_tensor_conversion_and_masked_pooling(self):
        import ast
        import torch
        from sklearn.metrics.pairwise import cosine_similarity

        source = Path(__file__).resolve().parents[1] / 'pipeline/03_sentence_embeddings.py'
        # Load the actual functions without executing the data/model-download stage.
        tree = ast.parse(source.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('mean_pooling', 'sent_embedding_centroid')]
        self.assertEqual(len(functions), 2)
        tokens = torch.tensor([[[1., 0.], [100., 100.]],
                               [[0., 1.], [100., 100.]],
                               [[1., 1.], [100., 100.]]])
        mask = torch.tensor([[1, 0], [1, 0], [1, 0]])
        namespace = dict(np=np, torch=torch, cosine_similarity=cosine_similarity,
                         sbert_tokenizer=lambda sentences, **kwargs: {'attention_mask': mask},
                         model=lambda **kwargs: (tokens,))
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
        centroid, dispersion, distance = namespace['sent_embedding_centroid'](['one', 'two', 'three'])
        self.assertIsInstance(centroid, np.ndarray)
        np.testing.assert_allclose(centroid, [2/3, 2/3], rtol=1e-6)
        self.assertAlmostEqual(float(dispersion), (2 / np.sqrt(2) + 1) / 3, places=6)
        self.assertAlmostEqual(float(distance), np.sqrt(2) + 1, places=6)


if __name__ == '__main__':
    unittest.main()
