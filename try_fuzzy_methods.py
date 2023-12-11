from src.methods.fuzzy.fuzzy_mamdani_inference import inference_mamdani_boosting_coeff, \
    inference_simple_mamdani_cb_mixing

if __name__ == '__main__':
    similarity = 0.85
    freshness = 3
    print(inference_mamdani_boosting_coeff(similarity, freshness))
    days = 1000
    print(inference_simple_mamdani_cb_mixing(similarity, days, 'terms_frequencies'))
