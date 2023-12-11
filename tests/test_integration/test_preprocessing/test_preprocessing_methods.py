import os

from src.prefillers.preprocessing.stopwords_loading import get_cz_stopwords_file_path, \
    get_general_stopwords_file_path


# python -m pytest .\tests\test_preprocessing\test_preprocessing_methods.py
# pytest.mark.integration
def test_if_stopwords_file_exists():
    assert os.path.exists(get_cz_stopwords_file_path())
    assert os.path.isfile(get_cz_stopwords_file_path())
    assert os.path.exists(get_general_stopwords_file_path())
    assert os.path.isfile(get_general_stopwords_file_path())

