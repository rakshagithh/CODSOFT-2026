from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    if request.method == "POST":
        data = [
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"])
        ]
        result = model.predict([data])[0]

        species = ["Setosa","Versicolor","Virginica"]
        prediction = species[int(result)]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
