import traceback

from src.prefillers.prefiller import prefilling_job_content_based


def prefill_doc2vec():
    """
    Invoking the Doc2Vec prefilling.
    :return:
    """
    while True:
        try:
            prefilling_job_content_based("doc2vec", full_text=False)
        except Exception as e:
            traceback.print_exception(None, e, e.__traceback__)


if __name__ == '__main__':
    prefill_doc2vec()
