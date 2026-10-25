from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("model.joblib")
target_names = ['setosa', 'versicolor', 'virginica']

@app.route("/", methods=["GET", "POST"])
def index():
    predict, result = None, None
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
        result = target_names[predict]

    return render_template("index.html", predict=predict, result=result)

if __name__ == "__main__":
    app.run(debug=True)
