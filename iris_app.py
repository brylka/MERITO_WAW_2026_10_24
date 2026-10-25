from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("model.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    predict = None
    if request.method == "POST":
        sl = float(request.form["sl"])
        sw = float(request.form["sw"])
        pl = float(request.form["pl"])
        pw = float(request.form["pw"])
        # print(sl,sw,pl,pw)
        sample = [[sl, sw, pl, pw]]
        # print(sample)
        predict = model.predict(sample)[0]
        # print(predict)

    return render_template("index.html", predict=predict)

if __name__ == "__main__":
    app.run(debug=True)
