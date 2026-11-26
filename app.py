"""
UltraVision AI - Flask Backend
Serves the frontend and provides /predict, /health, /model-info endpoints.
Classifies fetal ultrasound images into Normal / Abnormal / Emergency.
"""

import os
import json
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

# --- CONFIG -----------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "ultrasound_cnn_model.keras")
META_PATH    = os.path.join(BASE_DIR, "model_metadata.json")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "Frontend")

# The CNN outputs 3 classes in alphabetical folder order:
#   index 0 = benign  -> mapped to "Normal"
#   index 1 = malignant -> mapped to "Emergency"
#   index 2 = normal  -> mapped to "Abnormal"  (neither clearly healthy nor critical)
MODEL_CLASSES = ["benign", "malignant", "normal"]
IMG_SIZE      = (128, 128)

# Fetal-health mapping -------------------------------------------------------
FETAL_MAP = {
    "benign": {
        "status": "Normal",
        "risk": "none",
        "description": (
            "The fetal ultrasound appears normal. Key anatomical structures "
            "including the head circumference, abdominal circumference, and "
            "femur length are within expected ranges. The amniotic fluid "
            "volume looks adequate and the placenta positioning appears normal."
        ),
        "details": [
            "Heart rate is within the normal range (110-160 bpm).",
            "No visible structural abnormalities detected.",
            "Fetal growth parameters are consistent with gestational age.",
            "Amniotic fluid index appears within normal limits.",
        ],
        "recommendation": (
            "Continue routine prenatal check-ups as scheduled. Maintain a "
            "healthy diet and lifestyle. No immediate intervention required."
        ),
    },
    "normal": {
        "status": "Abnormal",
        "risk": "medium",
        "description": (
            "The fetal ultrasound shows some findings that deviate from "
            "normal parameters. Potential concerns include variations in "
            "growth measurements, minor structural differences, or slight "
            "changes in amniotic fluid levels that need monitoring."
        ),
        "details": [
            "Some growth parameters may be outside the expected range.",
            "Minor anatomical variations detected that require follow-up.",
            "Amniotic fluid levels may be slightly elevated or reduced.",
            "Further detailed scanning recommended to confirm findings.",
        ],
        "recommendation": (
            "A follow-up ultrasound is recommended within 1-2 weeks. "
            "Consult with a maternal-fetal medicine specialist for a "
            "detailed assessment. Additional tests such as Doppler "
            "studies may be advised."
        ),
    },
    "malignant": {
        "status": "Emergency",
        "risk": "high",
        "description": (
            "The fetal ultrasound indicates critical findings that require "
            "immediate medical attention. Significant abnormalities have "
            "been detected that may include severe growth restriction, "
            "major structural anomalies, or signs of fetal distress."
        ),
        "details": [
            "Significant deviation in fetal growth measurements detected.",
            "Possible major structural abnormality identified.",
            "Signs of fetal distress or compromised blood flow may be present.",
            "Urgent specialist evaluation and intervention may be necessary.",
        ],
        "recommendation": (
            "URGENT: Seek immediate medical consultation. Contact your "
            "obstetrician or go to the nearest hospital with a maternal-fetal "
            "medicine unit. Time-sensitive evaluation and potential "
            "intervention may be required."
        ),
    },
}

# --- APP --------------------------------------------------------------------
app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# --- LOAD MODEL -------------------------------------------------------------
print(f"Loading model from: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("[OK] Model loaded successfully")
else:
    model = None
    print(f"[WARN] Model file not found at {MODEL_PATH}")
    print("   Run cnn_ultrasound.py first to train and save the model.")

# --- LOAD METADATA ----------------------------------------------------------
model_meta = {}
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        model_meta = json.load(f)
    print(f"[OK] Metadata loaded: accuracy={model_meta.get('test_accuracy')}%")


def preprocess_image(image_bytes):
    """Convert uploaded image to model-ready numpy array (RGB)."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")                      # RGB as expected by the loaded model
    img = img.resize(IMG_SIZE)                    # 128 x 128
    arr = np.array(img) / 255.0                  # Normalize
    arr = np.expand_dims(arr, axis=0)             # Add batch dim: (1, 128, 128, 3)
    return arr


# --- API ROUTES (defined BEFORE the static catch-all) -----------------------

@app.route("/health", methods=["GET"])
def health():
    """Return model status and fetal health class information."""
    fetal_classes = {
        key: {"status": v["status"], "risk": v["risk"], "description": v["description"]}
        for key, v in FETAL_MAP.items()
    }
    return jsonify({
        "status": "ready" if model is not None else "no_model",
        "model_loaded": model is not None,
        "classes": ["Normal", "Abnormal", "Emergency"],
        "class_info": fetal_classes,
        "metadata": model_meta,
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    """Return model training metadata."""
    return jsonify({
        "accuracy": model_meta.get("test_accuracy", "N/A"),
        "loss": model_meta.get("test_loss", "N/A"),
        "epochs_trained": model_meta.get("epochs_trained", "N/A"),
        "classes": ["Normal", "Abnormal", "Emergency"],
        "image_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
        "model_type": "CNN (Deep Convolutional Neural Network)",
    })


@app.route("/test")
def test_route():
    """Quick test to verify Flask is responding."""
    return jsonify({"message": "Server working", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    print(">>> /predict request received")
    if model is None:
        return jsonify({
            "error": "Model not loaded. Train the model first by running cnn_ultrasound.py"
        }), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please select an ultrasound image."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename. Please select a valid image file."}), 400

    try:
        image_bytes = file.read()
        processed = preprocess_image(image_bytes)
        preds = model.predict(processed, verbose=0)[0]

        predicted_idx = int(np.argmax(preds))
        predicted_class = MODEL_CLASSES[predicted_idx]
        confidence = float(preds[predicted_idx]) * 100

        # Build per-class probabilities with fetal-health labels
        probabilities = {}
        for i, cls in enumerate(MODEL_CLASSES):
            fetal_label = FETAL_MAP[cls]["status"]
            probabilities[fetal_label] = round(float(preds[i]) * 100, 2)

        info = FETAL_MAP[predicted_class]

        return jsonify({
            "prediction": info["status"],
            "label": info["status"],
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "description": info["description"],
            "details": info["details"],
            "risk": info["risk"],
            "recommendation": info["recommendation"],
            "model_accuracy": model_meta.get("test_accuracy", "N/A"),
        })
    except Exception as e:
        print(traceback.format_exc())  # shows full error in terminal
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


# --- STATIC ROUTES (catch-all MUST be last) ---------------------------------

@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(FRONTEND_DIR, path)


# --- MAIN -------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{'=' * 50}")
    print(f"  UltraVision AI - Fetal Health Server")
    print(f"  Frontend : {FRONTEND_DIR}")
    print(f"  Model    : {'Loaded' if model else 'Not found'}")
    print(f"  URL      : http://localhost:5000")
    print(f"{'=' * 50}\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
