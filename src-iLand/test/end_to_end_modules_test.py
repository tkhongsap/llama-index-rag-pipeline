from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MODULES = ['data_processing', 'docs_embedding', 'load_embedding', 'retrieval']

def test_readme_files_exist():
    for module in MODULES:
        readme = BASE / module / 'README.md'
        assert readme.exists()
