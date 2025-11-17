import io
import json
import os
import sys
from typing import List

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import onnxruntime as ort

app = FastAPI(title="Model inference API")

# Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting FastAPI application...")

# Enable CORS for frontend (local y producción)
# En producción, permitir desde cualquier origen (Vercel, localhost, etc)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las orígenes
    allow_credentials=False,  # False cuando allow_origins es "*"
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

logger.info("CORS enabled for all origins")

# Try to locate model.onnx in the repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")

# Model registry: scan MODELS_DIR and load any folder containing model.onnx
MODELS = {}
DEFAULT_MODEL_ID = None

# Human-friendly names for known folders (override folder name or metadata)
MODEL_DISPLAY_MAP = {
    "model_1": "ConvNeXt",
    "model_2": "dlvtnet",
    "model_3": "mobilenetv3",
    "model_4": "swin_gsrdn",
}

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def load_models():
    global DEFAULT_MODEL_ID
    logger.info(f"Loading models from: {MODELS_DIR}")
    
    if not os.path.exists(MODELS_DIR):
        logger.warning(f"Models directory does not exist: {MODELS_DIR}")
        # fallback: look for model.onnx in repo root (legacy)
        legacy_model = os.path.join(ROOT, "model.onnx")
        if os.path.exists(legacy_model):
            logger.info(f"Found legacy model at {legacy_model}")
            sid = "model_1"
            sess = ort.InferenceSession(legacy_model, providers=["CPUExecutionProvider"])
            meta = sess.get_inputs()[0]
            shape = meta.shape
            H = safe_int(shape[2]) or 224
            W = safe_int(shape[3]) or 224
            MODELS[sid] = {
                "id": sid,
                "name": sid,
                "framework": "onnx",
                "session": sess,
                "input_name": meta.name,
                "input_shape": (H, W),
                "labels": [],
            }
            DEFAULT_MODEL_ID = sid
        else:
            logger.warning("No models found anywhere!")
        return

    for entry in sorted(os.listdir(MODELS_DIR)):
        model_folder = os.path.join(MODELS_DIR, entry)
        if not os.path.isdir(model_folder):
            continue
        # prefer model.onnx
        onnx_path = os.path.join(model_folder, "model.onnx")
        if not os.path.exists(onnx_path):
            logger.debug(f"No model.onnx found in {model_folder}")
            continue
        try:
            logger.info(f"Loading model: {entry}")
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            meta = sess.get_inputs()[0]
            shape = meta.shape
            H = safe_int(shape[2]) or 224
            W = safe_int(shape[3]) or 224
            # load labels from config.json or config_base.json if present
            labels = []
            for cfg_name in ("config.json", "config_base.json"):
                cfg_path = os.path.join(model_folder, cfg_name)
                if os.path.exists(cfg_path):
                    try:
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        if isinstance(cfg.get("labels"), list):
                            labels = cfg.get("labels")
                            break
                        if isinstance(cfg.get("classes"), list):
                            labels = cfg.get("classes")
                            break
                    except Exception:
                        labels = []

            # display name from metadata.json optional
            display_name = entry
            metadata_path = os.path.join(model_folder, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        md = json.load(f)
                        display_name = md.get("name") or display_name
                except Exception:
                    pass
            
            # override with our friendly names when available
            display_name = MODEL_DISPLAY_MAP.get(entry, display_name)

            MODELS[entry] = {
                "id": entry,
                "name": display_name,
                "framework": "onnx",
                "session": sess,
                "input_name": meta.name,
                "input_shape": (H, W),
                "labels": labels,
            }
            logger.info(f"✓ Loaded model: {entry} ({display_name})")
            if DEFAULT_MODEL_ID is None:
                DEFAULT_MODEL_ID = entry
        except Exception as e:
            # skip invalid model but keep going
            logger.error(f"✗ Failed to load model at {onnx_path}: {e}")
    
    logger.info(f"Total models loaded: {len(MODELS)}")
    logger.info(f"Default model: {DEFAULT_MODEL_ID}")


# load models at startup
try:
    load_models()
    logger.info("✓ Models loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load models: {e}")
    logger.error("Application will run but predictions may fail")

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup complete. Ready to serve requests.")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "models_loaded": len(MODELS),
        "default_model": DEFAULT_MODEL_ID
    }

@app.get("/debug")
def debug_info():
    """Debug endpoint para ver estado de modelos"""
    import os
    models_dir_exists = os.path.exists(MODELS_DIR)
    models_dir_contents = []
    
    if models_dir_exists:
        try:
            models_dir_contents = os.listdir(MODELS_DIR)
        except Exception as e:
            models_dir_contents = [f"Error: {e}"]
    
    return {
        "models_dir": MODELS_DIR,
        "models_dir_exists": models_dir_exists,
        "models_dir_contents": models_dir_contents,
        "models_loaded": len(MODELS),
        "models": {mid: {"name": m.get("name"), "labels": len(m.get("labels", []))} for mid, m in MODELS.items()},
        "default_model": DEFAULT_MODEL_ID
    }

def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    # Convert to RGB, resize, normalize to float32, CHW
    image = image.convert("RGB")
    image = image.resize(target_size, Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    # default mean/std (ImageNet) — these are sensible defaults for ConvNeXt-style models
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

@app.get("/labels")
def get_labels():
    # return labels for default model
    if DEFAULT_MODEL_ID and DEFAULT_MODEL_ID in MODELS:
        return {"model_id": DEFAULT_MODEL_ID, "labels": MODELS[DEFAULT_MODEL_ID].get("labels", [])}
    return {"labels": None, "note": "No labels found; no models loaded."}


@app.get("/favicon.ico")
def favicon():
    """Return empty favicon response so CRA dev-server proxy requests don't fail with ECONNREFUSED."""
    return Response(status_code=204)


@app.get("/models")
def list_models():
    # Return list of available models with basic info (no session objects)
    out = []
    for mid, m in MODELS.items():
        out.append({
            "id": mid,
            "name": m.get("name"),
            "framework": m.get("framework"),
            "num_labels": len(m.get("labels", [])),
            "labels": m.get("labels", []),
        })
    return {"models": out, "default": DEFAULT_MODEL_ID}


@app.post("/predict")
async def predict(model_id: str = None, file: UploadFile = File(...), top_k: int = 4):
    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Invalid image file", "detail": str(e)})
    # choose model
    chosen = model_id or DEFAULT_MODEL_ID
    if not chosen or chosen not in MODELS:
        return JSONResponse(status_code=400, content={"error": "Invalid or missing model_id", "available_models": list(MODELS.keys())})

    model = MODELS[chosen]
    H, W = model.get("input_shape", (224, 224))
    input_name = model.get("input_name")
    session = model.get("session")

    arr = preprocess_image(image, target_size=(H or 224, W or 224))
    # ONNX expects float32
    arr = arr.astype(np.float32)
    inputs = {input_name: arr}
    preds = session.run(None, inputs)[0]
    # preds shape: (1, num_classes) or similar
    if preds.ndim == 2:
        probs = softmax(preds[0])
    elif preds.ndim == 1:
        probs = softmax(preds)
    else:
        # flatten and softmax
        probs = softmax(preds.ravel())

    top_idx = np.argsort(probs)[-top_k:][::-1]
    results = []
    labels = model.get("labels", [])
    for idx in top_idx:
        label = labels[idx] if labels and idx < len(labels) else str(int(idx))
        results.append({"label": label, "index": int(idx), "score": float(probs[idx])})

    return {"predictions": results}
