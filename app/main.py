from flask import request
from flask_api import FlaskAPI
from sklearn.externals import joblib
import sys
from app.prediction import get_probabilities

app = FlaskAPI(__name__)


@app.route("/probabilities", methods=['POST'])
def probabilities():
    """
    For each document, return the probability that it is in category "1" in of the binary text classifier.
    """
    ids, texts = zip(*[(doc.get("id", "UNKNOWN_ID"), doc.get("text", ""))
                       for doc in request.data.get("documents", [])])
    print(texts)
    probabilities = [dict(id=id, probability=p) for id, p in zip(ids, get_probabilities(texts, model, vectorizer))]
    return {"probabilities": probabilities}

if __name__ == "__main__":
    modelDir = sys.argv[1]
    modelPath = modelDir + "lpi_LR_model.pkl"
    vectorizerPath = modelDir + "lpi_LR_vectorizer.pkl"

    model = joblib.load(modelPath)
    vectorizer = joblib.load(vectorizerPath)
    app.run(debug=True)