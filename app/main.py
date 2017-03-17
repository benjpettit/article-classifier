from flask import request
from flask_api import FlaskAPI
import pickle
from sklearn.externals import joblib
import sys
from string import punctuation

app = FlaskAPI(__name__)


@app.route("/probabilities", methods=['POST'])
def probabilities():
    """
    Classify text.
    """
    ids, texts = zip(*[(doc.get("id", "UNKNOWN_ID"), doc.get("text", ""))
                       for doc in request.data.get("documents", [])])
    print(texts)
    probabilities = [dict(id=id, probability=p) for id, p in zip(ids, get_probabilities(texts))]
    return {"probabilities": probabilities}

def get_probabilities(text_array):
    return model.predict_proba(vectorizer.transform([normalise_string(str(text)) for text in text_array]))[:, 1]

def get_probability(text):
    return model.predict_proba(vectorizer.transform([normalise_string(text)]))[0,1]

def normalise_string(string):
    return rm_punctuation(string.strip().lower())

def rm_punctuation(string, replacement='', exclude="'-'"):
    """Remove punctuation from an input string """
    string = string.replace('-', ' ')  # Always replace hyphen with space
    for p in set(list(punctuation)) - set(list(exclude)):
        string = string.replace(p, replacement)

    string = ' '.join(string.split())  # Remove excess whitespace
    return string

if __name__ == "__main__":
    # modelPath = sys.argv[0]
    # vectorizerPath = sys.argv[1]
    dir = "/Users/pettitb/Documents/biodiversity_hack/text_prediction"
    modelPath = dir + "/models/lpi_LR_model.pkl"
    vectorizerPath = dir + "/models/lpi_LR_vectorizer.pkl"

    model = joblib.load(modelPath)
    vectorizer = joblib.load(vectorizerPath)
    app.run(debug=True)