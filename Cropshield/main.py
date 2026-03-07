from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import io
import time

# ── Load model and class names ─────────────────────────────
session     = ort.InferenceSession("cropshield.onnx")
input_name  = session.get_inputs()[0].name

with open("class_names.json") as f:
    class_names = json.load(f)

# ── Disease treatment database ─────────────────────────────
TREATMENTS = {
    "healthy": "Your plant is healthy! Keep up good watering and sunlight.",
    "Apple_scab": "Apply fungicide containing captan or myclobutanil. Remove infected leaves.",
    "Apple___Black_rot": "Prune infected branches. Apply copper-based fungicide.",
    "Tomato___Late_blight": "Apply chlorothalonil fungicide. Remove infected plants immediately.",
    "Tomato___Early_blight": "Apply mancozeb fungicide. Ensure proper plant spacing.",
    "Corn_(maize)___Common_rust_": "Apply fungicide at early stages. Use resistant varieties.",
    "Potato___Late_blight": "Apply fungicide immediately. Remove and destroy infected plants.",
}

def get_treatment(class_name: str) -> str:
    for key in TREATMENTS:
        if key.lower() in class_name.lower():
            return TREATMENTS[key]
    if "healthy" in class_name.lower():
        return TREATMENTS["healthy"]
    return "Consult a local agricultural expert for treatment advice."

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr   = np.array(image).astype(np.float32) / 255.0
    mean  = np.array([0.485, 0.456, 0.406])
    std   = np.array([0.229, 0.224, 0.225])
    arr   = (arr - mean) / std
    arr   = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...].astype(np.float32)

# ── FastAPI App ────────────────────────────────────────────
app = FastAPI(
    title="CropShield API",
    description="Real-time plant disease detection — 38 classes, 99.64% accuracy",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "name":        "CropShield — Plant Disease Detection API",
        "version":     "1.0.0",
        "accuracy":    "99.64%",
        "classes":     38,
        "model":       "EfficientNetB0 + ONNX",
        "inference":   "~21ms",
        "endpoints":   ["/predict", "/health", "/classes", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded", "classes": len(class_names)}

@app.get("/classes")
def get_classes():
    return {"total": len(class_names), "classes": class_names}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, "Only JPEG/PNG images allowed")

    try:
        # Read and preprocess image
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents))
        input_tensor = preprocess_image(image)

        # Run inference
        start   = time.time()
        outputs = session.run(None, {input_name: input_tensor})[0]
        latency = (time.time() - start) * 1000

        # Get predictions
        probs      = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]))
        top5_idx   = np.argsort(probs)[::-1][:5]

        predicted_class = class_names[top5_idx[0]]
        confidence      = float(probs[top5_idx[0]])

        # Parse plant and disease
        parts     = predicted_class.split("___")
        plant     = parts[0].replace("_", " ") if len(parts) > 0 else predicted_class
        condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

        return {
            "predicted_class": predicted_class,
            "plant":           plant,
            "condition":       condition,
            "confidence":      round(confidence * 100, 2),
            "is_healthy":      "healthy" in predicted_class.lower(),
            "treatment":       get_treatment(predicted_class),
            "inference_ms":    round(latency, 2),
            "top5": [
                {
                    "class":      class_names[i],
                    "confidence": round(float(probs[i]) * 100, 2)
                }
                for i in top5_idx
            ]
        }

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)