from flask import Flask, render_template, request
from predictive_model import predict_for_product, predict_from_csv
import os
import logging

# Configure logging (INFO by default; set to DEBUG for detailed request logging)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ------------------------------------------------------------
# Flask App Configuration
# ------------------------------------------------------------
app = Flask(__name__)

# Ensure uploads folder exists
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ------------------------------------------------------------
# Home Route: Predict via Product ID
# ------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """
    Home page:
    - GET: Render prediction form.
    - POST: Retrieve product_id, forecast mode, and steps from form,
      run prediction using database data, and render results.
    """
    prediction_result = None

    if request.method == "POST":
        product_id = request.form["product_id"].strip()
        mode = request.form.get("mode", "daily")
        steps = int(request.form.get("steps", 1))

        logging.info(f"Predicting for Product ID={product_id}, Mode={mode}, Steps={steps}")
        prediction_result = predict_for_product(product_id, mode=mode, steps=steps)

    return render_template("index.html", prediction=prediction_result)


# ------------------------------------------------------------
# Upload Route: Predict via Uploaded CSV
# ------------------------------------------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    """
    Upload page:
    - GET: Show upload form.
    - POST: Validate and save CSV, run prediction on uploaded data.
    """
    prediction_result = None

    if request.method == "POST":
        uploaded_file = request.files.get("file")
        mode = request.form.get("mode", "daily")
        steps = int(request.form.get("steps", 1))

        # Validate file type and run prediction
        if uploaded_file and uploaded_file.filename.endswith(".csv"):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
            uploaded_file.save(filepath)

            logging.info(f"Uploaded CSV saved at: {filepath}")
            prediction_result = predict_from_csv(filepath, mode=mode, steps=steps)

    return render_template("upload.html", prediction=prediction_result)


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
