import os
import random
from unittest import TestCase
from unittest.mock import call

from unittest.mock import patch

import pytest

from src.custom_exceptions.exceptions import TestRunException
from src.data_handling.data_manipulation import DatabaseMethods
from src.prefillers.prefiller import UserBased
from src.prefillers.user_based_prefillers.prefilling_collaborative import run_prefilling_collaborative

database = DatabaseMethods()
method_options_short_text = ["terms_frequencies", "word2vec", "doc2vec", "topics"]
method_options_full_text = ["terms_frequencies", "word2vec", "doc2vec", "topics",
                            "word2vec_eval_idnes_1", "word2vec_eval_idnes_2",
                            "word2vec_eval_idnes_3", "word2vec_eval_idnes_4", "word2vec_eval_cswiki_1",
                            "doc2vec_eval_cswiki_1"]

full_text_options = [True, False]
random_reverse_options = [True, False]


class TestConnection:
    # python -m pytest .\tests\test_prefillers_methods.py::ConnectionTest::test_db_connection
    # pytest.mark.integration
    @patch("psycopg2.connect")
    def test_db_connection(self, mockconnect):
        # noinspection PyPep8Naming
        DB_USER = os.environ.get('DB_RECOMMENDER_USER')
        # noinspection PyPep8Naming
        DB_PASSWORD = os.environ.get('DB_RECOMMENDER_PASSWORD')
        # noinspection PyPep8Naming
        DB_HOST = os.environ.get('DB_RECOMMENDER_HOST')
        # noinspection PyPep8Naming
        DB_NAME = os.environ.get('DB_RECOMMENDER_NAME')

        assert type(DB_USER) is str
        assert type(DB_PASSWORD) is str
        assert type(DB_HOST) is str
        assert type(DB_NAME) is str

        assert bool(DB_USER) is True  # not empty
        assert bool(DB_PASSWORD) is True
        assert bool(DB_HOST) is True
        assert bool(DB_NAME) is True

        database.connect()
        mockconnect.assert_called()
        assert 1 == mockconnect.call_count
        assert mockconnect.call_args_list[0] == call(user=DB_USER, password=DB_PASSWORD,
                                                     host=DB_HOST, dbname=DB_NAME,
                                                     keepalives=1, keepalives_idle=30, keepalives_interval=5,
                                                     keepalives_count=5)


# python -m pytest .\tests\test_prefillers_methods.py::test_not_prefilled_retriaval
# pytest.mark.integration
def not_prefilled_retriaval(method):
    database_methods = DatabaseMethods()
    database_methods.connect()
    not_prefilled_posts = database_methods.get_not_prefilled_posts(method=method)
    database_methods.disconnect()
    return isinstance(not_prefilled_posts, list)


# pytest.mark.integration
class TestPrefillers:
    # pytest.mark.integration
    def test_prefillers(self):
        for i in range(2):
            random_method_choice = random.choice(method_options_short_text)
            assert not_prefilled_retriaval(method=random_method_choice) \
                   is True

            random_method_choice = random.choice(method_options_full_text)
            assert not_prefilled_retriaval(method=random_method_choice) \
                   is True


# pytest.mark.integration
class TestUserPrefillers(TestCase):

    def test_user_preferences_prefiller(self):
        with pytest.raises(TestRunException):
            run_prefilling_collaborative(test_run=True)

    @patch.object(UserBased, "prefilling_job_user_based", autospec=UserBased)
    def test_prefilling_job_user_based_not_called(self, mock_prefilling_job_user_based):
        methods = ['svd', 'user_keywords', 'best_rated', 'hybrid']  # last value is BS value
        with pytest.raises(ValueError):
            run_prefilling_collaborative(methods)
        mock_prefilling_job_user_based.assert_not_called()
