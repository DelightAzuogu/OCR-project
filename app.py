import os
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
import io

# Define lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model and mappings
    global model, char_mappings

    # Load model
    model_path = "saved_model11"
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}")

    try:
        model = tf.saved_model.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

    # Load character mappings
    mapping_path = "char_mappings.json"
    if not os.path.exists(mapping_path):
        raise RuntimeError(f"Character mappings not found at {mapping_path}")

    try:
        with open(mapping_path, 'r') as f:
            char_mappings = json.load(f)
        print("Character mappings loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load character mappings: {str(e)}")

    yield

    # Shutdown: Clean up resources if needed
    # No specific cleanup needed for this application

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Character Recognition API",
    description="API for recognizing characters from images using a trained TensorFlow model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for model and mappings
model = None
char_mappings = None
IMG_HEIGHT = 60
IMG_WIDTH = 40

def preprocess_image(image_bytes):
    """Preprocess the image bytes to match the model's input requirements"""
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to grayscale if it's not already
        if img.mode != 'L':
            img = img.convert('L')

        # Resize to match model input size
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))

        # Convert to numpy array and normalize - EXPLICITLY USE float32
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch and channel dimensions
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

        return img_array

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/predict/", response_class=JSONResponse)
async def predict_character(file: UploadFile = File(...)):
    """Endpoint to predict character from uploaded image"""
    # Check if model is loaded
    if model is None or char_mappings is None:
        raise HTTPException(status_code=503, detail="Model or character mappings not loaded")

    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read file contents
        contents = await file.read()

        # Preprocess image
        input_tensor = preprocess_image(contents)

        # Ensure correct data type (float32)
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

        # Get prediction function from the model
        infer = model.signatures["serving_default"]

        # Make prediction
        predictions = infer(input_tensor)

        # Get output tensor name (might vary based on your model)
        output_key = list(predictions.keys())[0]
        prediction = predictions[output_key].numpy()

        # Get the predicted class index
        predicted_idx = np.argmax(prediction[0])

        # Convert to character using mappings
        num_to_char = char_mappings.get("num_to_char", {})
        # Convert integer index to string for JSON key lookup
        predicted_char = num_to_char.get(str(predicted_idx), "Unknown")

        # Get confidence score
        confidence = float(prediction[0][predicted_idx])

        return {
            "predicted_character": predicted_char,
            "confidence": confidence,
            "class_index": int(predicted_idx)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health/")
async def health_check():
    """Endpoint to check if the API is running and model is loaded"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "char_mappings_loaded": char_mappings is not None
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Character Recognition API",
        "usage": "POST an image to /predict/ endpoint",
        "health_check": "/health/"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)