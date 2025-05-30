from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model once at startup
try:
    # In Azure, the model will be in the wwwroot directory
    model_path = os.path.join(os.getcwd(), "wand_model.h5")
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.route("/", methods=["GET"])
def home():
    return "Wand Gesture API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get and validate input data
        data = request.json.get("data")
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Convert to numpy array and validate shape
        input_array = np.array(data)
        expected_shape = (None, 100, 3)  # Adjust based on your model's expected input shape
        if input_array.shape[1:] != expected_shape[1:]:
            return jsonify({
                "error": f"Invalid input shape. Expected {expected_shape[1:]}, got {input_array.shape[1:]}"
            }), 400
            
        # Make prediction
        logger.info(f"Making prediction for input shape: {input_array.shape}")
        prediction = model.predict(input_array)
        
        # Convert prediction to gesture labels
        gesture_classes = ["O", "V"]
        predicted_class = gesture_classes[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence,
            "raw_prediction": prediction.tolist()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
    