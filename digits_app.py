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
        file = request.files["image"]           # wczytaj plik z formularza
        image = Image.open(file)                # otwórz wczytany plik
        image = image.convert("L")              # zamień na skalę szarości
        image = image.resize((8,8))             # zmniejsz rozmiar na 8x8
        pixels = np.array(image, dtype=float)   # zamień obraz na tablicę liczbową
        pixels = 255 - pixels                   # odwróć skalę 0-255 -> 255-0
        pixels = pixels / 255 * 16              # spłaszcz skalę do 0-16
        vector = pixels.flatten()               # tablica 8x8 na wektor liczb
        sample = [vector]                       # dodanie do listy
        predict = model.predict(sample)[0]      # predykcja

    return render_template("digits.html", predict=predict)

if __name__ == "__main__":
    app.run(debug=True)
