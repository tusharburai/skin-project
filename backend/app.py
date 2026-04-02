from backend.predict import predict_image
from flask import Flask, request, jsonify, render_template
from backend.predict import predict_image
import os

app = Flask(__name__, template_folder="../templates", static_folder="../static")

UPLOAD_FOLDER = "../static/uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    result = predict_image(file_path)

    return jsonify({
        "prediction": result,
        "image": file.filename
    })

if __name__ == "__main__":
    app.run(debug=True)