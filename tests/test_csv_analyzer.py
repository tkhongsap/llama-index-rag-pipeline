import importlib.util
from pathlib import Path
import pytest

pd = pytest.importorskip("pandas")

# Load modules from src-iLand/data_processing
base = Path(__file__).resolve().parents[1] / 'src-iLand' / 'data_processing'

models_spec = importlib.util.spec_from_file_location('models', base / 'models.py')
models = importlib.util.module_from_spec(models_spec)
import sys
models_spec.loader.exec_module(models)
sys.modules['models'] = models

mod_spec = importlib.util.spec_from_file_location('csv_analyzer', base / 'csv_analyzer.py')
csv_analyzer = importlib.util.module_from_spec(mod_spec)
mod_spec.loader.exec_module(csv_analyzer)


def test_infer_be_date_slash():
    analyzer = csv_analyzer.CSVAnalyzer()
    series = pd.Series(['25/09/2567', '01/01/2568', '15/12/2567'])
    assert analyzer._infer_data_type(series) == 'date'


def test_infer_be_date_dash():
    analyzer = csv_analyzer.CSVAnalyzer()
    series = pd.Series(['2567-09-25', '2567-12-15', '2568-01-01'])
    assert analyzer._infer_data_type(series) == 'date'
