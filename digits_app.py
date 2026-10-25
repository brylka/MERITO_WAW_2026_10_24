from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("model.joblib")


@app.route("/", methods=["GET", "POST"])
def index():
    predict = None
    if request.method == "POST":
        pass


    return render_template("digits.html", predict=predict)

if __name__ == "__main__":
    app.run(debug=True)
