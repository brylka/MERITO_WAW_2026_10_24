from PIL import Image
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.joblib")


@app.route("/", methods=["GET", "POST"])
def index():
    predict = None
    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file)
        image = image.convert("L")
        image = image.resize((8,8))
        pixels = np.array(image, dtype=float)
        pixels = 255 - pixels
        pixels = pixels / 255 * 16
        vector = pixels.flatten()
        sample = [vector]
        predict = model.predict(sample)[0]


    return render_template("digits.html", predict=predict)

if __name__ == "__main__":
    app.run(debug=True)
