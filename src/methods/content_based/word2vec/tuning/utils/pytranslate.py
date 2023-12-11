import logging

from deep_translator import GoogleTranslator, exceptions


def translate_question_words():
    texts = []
    already_processed_texts = []
    with (open('../../../../../../stats/evaluations/word2vec/analogies/questions-words.txt', 'redis_instance', encoding="utf-8")
          as file):
        texts.extend(file.read().split("\n"))

    with (open('../../../../../../stats/evaluations/word2vec/analogies/questions-words-cs.txt', 'redis_instance', encoding="utf-8")
          as file):
        already_processed_texts.extend(file.read().split("\n"))

    num_of_already_processed = len(already_processed_texts)

    logging.info("TRANSLATING TEXTS...")
    translations = []
    batch_size = 100
    batch_counter = 0
    i = 0
    for text in texts:
        if i < num_of_already_processed:
            logging.info("Skipping already translated.")
            pass
        else:
            try:
                translation = GoogleTranslator(source='en', target='cs').translate(text)
                translations.append(translation + "\n")
                if batch_counter == batch_size:
                    with open('../../../../../../stats/evaluations/word2vec/analogies/questions-words-cs.txt', 'a',
                              encoding="utf-8") as file:
                        file.writelines(translations)
                        batch_counter = 0
                        translations = []
                batch_counter = batch_counter + 1
            except exceptions.TranslatorException as e:
                print("Translation error: ", e)

        i = i + 1


def google_translate(text_to_translate):
    translation = GoogleTranslator(source='en', target='cs').translate(text_to_translate)
    return translation


def clean_console_output_to_file():

    texts = []
    with open('../../../../../../stats/evaluations/word2vec/translations/questions-words-cs-console-copy.txt', 'redis_instance',
              encoding="utf-8") as file:
        texts.extend(file.read().split("\n"))
    print("text:")
    print(texts)
    texts_cleaned = []
    i = 1

    for line in texts:
        # Translation occurs in every 3rd line in current output in the format:
        """
        INPUT text:
        Athens Greece Baghdad Iraq
        translation:
        Atény Řecko Bagdád Irák
        """
        if i == 4:
            print("Adding line:")
            print(line)
            texts_cleaned.extend(line + "\n")
            i = 0
        i = i + 1

        with open('../../../../../../stats/evaluations/word2vec/translations/questions-words-cs-translated_so_far.txt',
                  'w+', encoding="utf-8") as file:
            file.writelines(texts_cleaned)


def run_clean_console_output_to_file():
    clean_console_output_to_file()


def run_translation_of_question_words():
    translate_question_words()
