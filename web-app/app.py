import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("./ufo-model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    #countries = ["Australia", "Canada", "Germany", "UK", "US"]
    shape = ['cylinder', 'light',  'circle',  'sphere',  'disk',  'fireball',  'unknown',  'oval', 
 'other',  'cigar',  'rectangle',  'chevron',  'triangle',  'formation', 'delta', 
 'changing',  'egg',  'diamond',  'flash',  'teardrop',  'cone',  'cross',  'pyramid', 
 'round',  'crescent',  'flare',  'hexagon',  'dome',  'changed']
    return render_template(
        "index.html", prediction_text="The most likely shape is: {}".format(shape[output])
    )


if __name__ == "__main__":
    app.run(debug=True)