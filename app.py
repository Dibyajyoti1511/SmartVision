from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io, os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model/cnn_model.h5"
CLASS_NAMES = ["cat", "dog"]

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB').resize((128,128))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error":"no file"}), 400
    f = request.files['file']
    img_arr = preprocess_image(f.read())
    preds = model.predict(img_arr)[0]
    idx = int(np.argmax(preds))
    result = {"class": CLASS_NAMES[idx], "confidence": float(preds[idx])}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
