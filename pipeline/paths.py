"""Generic local paths. Run the modules from the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXTS = str(ROOT / 'data' / 'texts') + '/'
METADATA = str(ROOT / 'data' / 'metadata') + '/'
SANTANDER = str(ROOT / 'data' / 'santander') + '/'
RESULTS = str(ROOT / 'results') + '/'
TABLES = str(ROOT / 'results' / 'tables') + '/'
ADDITIONAL_RESULTS = str(ROOT / 'results' / 'additional') + '/'
SUMMARIES = str(ROOT / 'results' / 'additional' / 'summaries') + '/'
NETWORK_RESULTS = str(ROOT / 'results' / 'additional' / 'umap') + '/'
