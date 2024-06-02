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
    shapes = ['Cylinder', 'Light', 'Circle', 'Sphere', 'Disk', 'Fireball', 'Unknown', 'Oval',
          'Other', 'Cigar', 'Rectangle', 'Chevron', 'Triangle', 'Formation', 'Delta',
          'Changing', 'Egg', 'Diamond', 'Flash', 'Teardrop', 'Cone', 'Cross', 'Pyramid',
          'Round', 'Crescent', 'Flare', 'Hexagon', 'Dome', 'Changed']
    return render_template(
        "index.html", prediction_text="The most likely shape is: {}".format(shapes[output])
    )


if __name__ == "__main__":
    app.run(debug=True)