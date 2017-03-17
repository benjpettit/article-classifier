from string import punctuation

def get_probabilities(text_array, model, vectorizer):
    return model.predict_proba(vectorizer.transform([normalise_string(str(text)) for text in text_array]))[:, 1]

def normalise_string(string):
    return rm_punctuation(string.strip().lower())

def rm_punctuation(string, replacement='', exclude="'-'"):
    """Remove punctuation from an input string """
    string = string.replace('-', ' ')  # Always replace hyphen with space
    for p in set(list(punctuation)) - set(list(exclude)):
        string = string.replace(p, replacement)

    string = ' '.join(string.split())  # Remove excess whitespace
    return string