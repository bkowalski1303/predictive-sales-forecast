from flask import Flask, render_template, request, redirect, url_for
from predictive_model import predict_for_product, predict_from_csv 
import os

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    if request.method == "POST":
        product_id = request.form["product_id"].strip()
        mode = request.form.get("mode", "daily")
        steps = int(request.form.get("steps", 1)) 
        result = predict_for_product(product_id, mode=mode, steps=steps)
        prediction_result = result
    return render_template("index.html", prediction=prediction_result)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    prediction_result = None
    if request.method == "POST":
        file = request.files["file"]
        mode = request.form.get("mode", "daily")
        steps = int(request.form.get("steps", 1))  
        if file and file.filename.endswith(".csv"):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            prediction_result = predict_from_csv(filepath, mode=mode, steps=steps)
    return render_template("upload.html", prediction=prediction_result)

if __name__ == "__main__":
    app.run(debug=True)
