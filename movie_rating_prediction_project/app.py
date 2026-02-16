from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    if request.method == "POST":
        data = [
            int(request.form["genre"]),
            int(request.form["director"]),
            int(request.form["actor1"]),
            int(request.form["actor2"]),
            int(request.form["actor3"])
        ]
        prediction = model.predict([data])[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
