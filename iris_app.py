from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        sl = float(request.form["sl"])
        sw = float(request.form["sw"])
        pl = float(request.form["pl"])
        pw = float(request.form["pw"])
        # print(sl,sw,pl,pw)
        sample = [[sl, sw, pl, pw]]
        print(sample)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
