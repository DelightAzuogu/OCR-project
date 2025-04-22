from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import io
from PIL import Image
import uvicorn
import os
import pandas as pd
import json

# === CONFIG ===
MODEL_PATH = "/best_model"  # Directory containing the SavedModel (PB format)
IMG_HEIGHT = 64
IMG_WIDTH = 64
TRAIN_CSV_PATH = "train.csv"
TEST_CSV_PATH = "dataset.csv"
CHAR_MAPPING_PATH = "char_mapping.json"  # Path to saved character mapping

# === GLOBALS ===
char_to_num = None
num_to_char = None
model = None

# === INIT FASTAPI ===
app = FastAPI(
    title="Character Recognition API",
    description="API for recognizing characters from uploaded images",
    version="1.0.0",
)

# === ALLOW CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === UTILITY FUNCTIONS ===
def load_character_mapping():
    """Load character mapping from JSON file or recreate from CSV"""
    try:
        # Try to load from saved JSON file first
        if os.path.exists(CHAR_MAPPING_PATH):
            with open(CHAR_MAPPING_PATH, 'r') as f:
                mapping_data = json.load(f)
                char_to_num = mapping_data['char_to_num']
                # Convert string keys back to integers for num_to_char
                num_to_char = {int(k): v for k, v in mapping_data['num_to_char'].items()}
                print(f"Character mapping loaded from JSON with {len(num_to_char)} characters")
                return char_to_num, num_to_char

        # Fallback to creating from CSV
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        test_df = pd.read_csv(TEST_CSV_PATH)

        all_chars = set(train_df['label'].unique()).union(set(test_df['label'].unique()))
        unique_chars = sorted(list(all_chars))

        char_to_num = {char: idx for idx, char in enumerate(unique_chars)}
        num_to_char = {idx: char for char, idx in char_to_num.items()}
        print(f"Character mapping created from CSV with {len(num_to_char)} characters")
        return char_to_num, num_to_char

    except Exception as e:
        print(f"[WARNING] Failed to load character mapping: {e}")
        fallback_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        char_to_num = {char: idx for idx, char in enumerate(fallback_chars)}
        num_to_char = {idx: char for idx, char in enumerate(fallback_chars)}
        print(f"Using fallback character mapping with {len(num_to_char)} characters")
        return char_to_num, num_to_char

def load_saved_model():
    """Load model from SavedModel directory"""
    try:
        print(f"Loading SavedModel from {MODEL_PATH}")
        model = tf.saved_model.load(MODEL_PATH)
        print("SavedModel loaded successfully")
        return model
    except Exception as e:
        error_msg = f"Failed to load SavedModel: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

def preprocess_image(image_data):
    """Process image for model input"""
    try:
        # Open and convert to grayscale
        img = Image.open(io.BytesIO(image_data)).convert('L')

        # Convert PIL Image to numpy array
        img_array = np.array(img)

        # Add channel dimension to meet the requirements of tf.image.resize_with_pad
        img_array = np.expand_dims(img_array, axis=-1)

        # Resize using TensorFlow
        img = tf.image.resize_with_pad(img_array, IMG_HEIGHT, IMG_WIDTH)
        img = tf.cast(img, tf.float32) / 255.0
        # No need to add channel dimension again since we already did
        img = tf.expand_dims(img, axis=0)   # Add batch dimension

        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

# === INITIALIZE RESOURCES AT STARTUP ===
@app.on_event("startup")
async def startup_event():
    """Initialize resources at startup"""
    global char_to_num, num_to_char, model

    # Load character mapping
    char_to_num, num_to_char = load_character_mapping()

    # Pre-load model to speed up first inference
    try:
        model = load_saved_model()
    except Exception as e:
        print(f"Warning: Could not pre-load model at startup: {e}")
        print("Will attempt to load model when needed for inference")

# === ROUTES ===
@app.get("/")
async def root():
    """Root endpoint with information about API"""
    return {
        "message": "Character Recognition API",
        "endpoints": {
            "GET /health": "Check if the API is alive",
            "POST /predict": "Upload an image and get character prediction"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global model

    # Attempt to load model if not already loaded
    if model is None:
        try:
            model = load_saved_model()
            model_status = "loaded"
        except:
            model_status = "not loaded"
    else:
        model_status = "loaded"

    return {
        "status": "healthy",
        "model": model_status,
        "char_mapping": "loaded" if num_to_char is not None else "not loaded"
    }

@app.post("/predict")
async def predict_character(file: UploadFile = File(...)):
    """Predict character from uploaded image"""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    try:
        # Ensure character mapping is loaded
        global char_to_num, num_to_char, model
        if char_to_num is None or num_to_char is None:
            char_to_num, num_to_char = load_character_mapping()

        # Load model if not already loaded
        if model is None:
            model = load_saved_model()

        # Read and process image
        contents = await file.read()
        processed_image = preprocess_image(contents)

        # Make prediction with SavedModel
        try:
            # Try direct calling first (works with many SavedModels)
            predictions = model(processed_image)

            # Handle different return formats
            if isinstance(predictions, dict):  # SignatureDef output format
                # Get output tensor (usually the last one or "predictions")
                output_key = list(predictions.keys())[-1]
                predictions = predictions[output_key]
        except:
            # Try using the serving signature if direct call fails
            try:
                infer = model.signatures["serving_default"]
                result = infer(tf.constant(processed_image))
                # Get the output tensor (usually the last one)
                output_key = list(result.keys())[-1]
                predictions = result[output_key]
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get prediction from SavedModel: {str(e)}"
                )

        # Convert to numpy for processing if it's a tensor
        if isinstance(predictions, tf.Tensor):
            predictions = predictions.numpy()

        # Get predicted class
        if len(predictions.shape) == 2:  # [batch_size, num_classes]
            predicted_class_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class_idx])
        else:
            # Handle unexpected output shape
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected prediction shape: {predictions.shape}"
            )

        # Get character from index
        predicted_char = num_to_char.get(predicted_class_idx, "?")

        return {
            "predicted_character": predicted_char,
            "confidence": confidence,
            "class_index": predicted_class_idx
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# === RUN ===
if __name__ == "__main__":
    # Get the current file name without extension
    module_name = os.path.basename(__file__).replace('.py', '')
    uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=3000, reload=True)