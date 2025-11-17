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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

logger.info("CORS enabled for all origins")

# Try to locate model.onnx in the repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")

# Model registry: store loaded models (lazy loading)
MODELS = {}
DEFAULT_MODEL_ID = None
AVAILABLE_MODEL_IDS = []
MODEL_METADATA_CACHE = {}

# Human-friendly names for known folders
MODEL_DISPLAY_MAP = {
    "model_1": "ConvNeXt",
    "model_2": "dlvtnet",
    "model_3": "mobilenetv3",
    "model_4": "swin_gsrdn",
}

# Production environment flag
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"

# Memory management configuration
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "400"))
MODEL_MEMORY_ESTIMATES = {
    "model_1": 70,
    "model_2": 60,
    "model_3": 50,
    "model_4": 150,
}
MAX_CONCURRENT_MODELS = int(os.getenv("MAX_CONCURRENT_MODELS", "2"))
MODEL_USAGE_TRACKER = {}

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0

def estimate_total_model_memory():
    """Estimate total memory used by loaded models"""
    total = 0
    for model_id in MODELS:
        total += MODEL_MEMORY_ESTIMATES.get(model_id, 100)
    return total

def can_load_model(model_id: str):
    """Check if we can load a model without exceeding memory limits"""
    current_memory = estimate_total_model_memory()
    new_model_memory = MODEL_MEMORY_ESTIMATES.get(model_id, 100)
    
    if current_memory + new_model_memory > MAX_MEMORY_MB:
        return False
    
    if len(MODELS) >= MAX_CONCURRENT_MODELS and model_id not in MODELS:
        return False
        
    return True

def unload_least_recently_used_model():
    """Unload the least recently used model to free memory"""
    if not MODELS or not MODEL_USAGE_TRACKER:
        return None
        
    lru_model = min(MODEL_USAGE_TRACKER.items(), key=lambda x: x[1])
    model_id = lru_model[0]
    
    if model_id in MODELS:
        logger.info(f"🗑️  Unloading model {model_id} to free memory")
        del MODELS[model_id]
        del MODEL_USAGE_TRACKER[model_id]
        return model_id
    return None

def update_model_usage(model_id: str):
    """Update last usage time for a model"""
    import time
    MODEL_USAGE_TRACKER[model_id] = time.time()

def get_model_metadata(model_id: str):
    """Get metadata for a model (name, labels) without loading ONNX"""
    display_name = MODEL_DISPLAY_MAP.get(model_id, model_id)
    labels = []
    
    model_folder = os.path.join(MODELS_DIR, model_id)
    
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
            except Exception as e:
                logger.debug(f"Failed to read config for {model_id}: {e}")
    
    return {
        "id": model_id,
        "name": display_name,
        "framework": "onnx",
        "num_labels": len(labels),
        "labels": labels,
    }

def discover_models():
    """Scan MODELS_DIR and list available models WITHOUT loading them"""
    global DEFAULT_MODEL_ID, AVAILABLE_MODEL_IDS, MODEL_METADATA_CACHE
    logger.info(f"Scanning models directory: {MODELS_DIR}")
    logger.info(f"Environment: {'PRODUCTION' if IS_PRODUCTION else 'DEVELOPMENT'}")
    AVAILABLE_MODEL_IDS = []
    MODEL_METADATA_CACHE = {}
    
    if not os.path.exists(MODELS_DIR):
        logger.warning(f"Models directory does not exist: {MODELS_DIR}")
        return

    for entry in sorted(os.listdir(MODELS_DIR)):
        model_folder = os.path.join(MODELS_DIR, entry)
        if not os.path.isdir(model_folder):
            continue
        onnx_path = os.path.join(model_folder, "model.onnx")
        if os.path.exists(onnx_path):
            AVAILABLE_MODEL_IDS.append(entry)
            MODEL_METADATA_CACHE[entry] = get_model_metadata(entry)
            if DEFAULT_MODEL_ID is None:
                DEFAULT_MODEL_ID = entry
    
    logger.info(f"✓ Found {len(AVAILABLE_MODEL_IDS)} available models: {AVAILABLE_MODEL_IDS}")
    logger.info(f"Default model will be: {DEFAULT_MODEL_ID}")

def load_single_model(model_id: str):
    """Load a single model on-demand with smart memory management"""
    if model_id in MODELS:
        update_model_usage(model_id)
        return MODELS[model_id]
    
    if model_id not in AVAILABLE_MODEL_IDS:
        logger.error(f"Model {model_id} not found in available models")
        return None
    
    while not can_load_model(model_id) and len(MODELS) > 0:
        unloaded = unload_least_recently_used_model()
        if unloaded is None:
            break
    
    if not can_load_model(model_id):
        current_mem = estimate_total_model_memory()
        required_mem = MODEL_MEMORY_ESTIMATES.get(model_id, 100)
        logger.warning(f"⚠️  Cannot load {model_id}: would exceed memory limit ({current_mem + required_mem}MB > {MAX_MEMORY_MB}MB)")
        return None
    
    model_folder = os.path.join(MODELS_DIR, model_id)
    onnx_path = os.path.join(model_folder, "model.onnx")
    
    try:
        current_mem = estimate_total_model_memory()
        logger.info(f"🧠 Loading model: {model_id} (current memory: {current_mem}MB)")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        meta = sess.get_inputs()[0]
        shape = meta.shape
        H = safe_int(shape[2]) or 224
        W = safe_int(shape[3]) or 224
        
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

        display_name = model_id
        metadata_path = os.path.join(model_folder, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    md = json.load(f)
                    display_name = md.get("name") or display_name
            except Exception:
                pass
        
        display_name = MODEL_DISPLAY_MAP.get(model_id, display_name)

        MODELS[model_id] = {
            "id": model_id,
            "name": display_name,
            "framework": "onnx",
            "session": sess,
            "input_name": meta.name,
            "input_shape": (H, W),
            "labels": labels,
        }
        
        update_model_usage(model_id)
        
        new_mem = estimate_total_model_memory()
        logger.info(f"✅ Loaded model: {model_id} ({display_name}) - Total memory: {new_mem}MB")
        return MODELS[model_id]
    except Exception as e:
        logger.error(f"✗ Failed to load model {model_id}: {e}")
        return None


# Discover available models at startup
try:
    discover_models()
    logger.info("✓ Model discovery successful (lazy loading enabled)")
except Exception as e:
    logger.error(f"✗ Failed to discover models: {e}")
    logger.error("Application will run but model loading may fail")

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup complete. Ready to serve requests.")

@app.get("/health")
def health_check():
    """Health check endpoint with memory info"""
    current_memory = get_memory_usage()
    estimated_model_memory = estimate_total_model_memory()
    
    return {
        "status": "ok",
        "models_available": len(AVAILABLE_MODEL_IDS),
        "models_loaded": len(MODELS),
        "default_model": DEFAULT_MODEL_ID,
        "memory": {
            "current_mb": round(current_memory, 1),
            "estimated_models_mb": estimated_model_memory,
            "max_limit_mb": MAX_MEMORY_MB,
            "max_concurrent_models": MAX_CONCURRENT_MODELS
        },
        "loaded_models": list(MODELS.keys())
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
    
    current_memory = get_memory_usage()
    estimated_model_memory = estimate_total_model_memory()
    
    return {
        "models_dir": MODELS_DIR,
        "models_dir_exists": models_dir_exists,
        "models_dir_contents": models_dir_contents,
        "available_models": AVAILABLE_MODEL_IDS,
        "models_currently_loaded": len(MODELS),
        "models_loaded_details": {mid: {"name": m.get("name"), "labels": len(m.get("labels", []))} for mid, m in MODELS.items()},
        "default_model": DEFAULT_MODEL_ID,
        "memory_management": {
            "current_memory_mb": round(current_memory, 1),
            "estimated_model_memory_mb": estimated_model_memory,
            "max_memory_limit_mb": MAX_MEMORY_MB,
            "max_concurrent_models": MAX_CONCURRENT_MODELS,
            "model_memory_estimates": MODEL_MEMORY_ESTIMATES,
            "usage_tracker": MODEL_USAGE_TRACKER
        }
    }

def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(target_size, Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

@app.get("/labels")
def get_labels():
    if DEFAULT_MODEL_ID and DEFAULT_MODEL_ID in MODELS:
        return {"model_id": DEFAULT_MODEL_ID, "labels": MODELS[DEFAULT_MODEL_ID].get("labels", [])}
    return {"labels": None, "note": "No labels found; no models loaded."}

@app.get("/memory")
def memory_status():
    """Get current memory status and loaded models"""
    current_memory = get_memory_usage()
    estimated_model_memory = estimate_total_model_memory()
    
    return {
        "current_memory_mb": round(current_memory, 1),
        "estimated_model_memory_mb": estimated_model_memory,
        "max_memory_limit_mb": MAX_MEMORY_MB,
        "memory_usage_percent": round((estimated_model_memory / MAX_MEMORY_MB) * 100, 1),
        "loaded_models": list(MODELS.keys()),
        "can_load_more": len(MODELS) < MAX_CONCURRENT_MODELS,
        "model_memory_estimates": MODEL_MEMORY_ESTIMATES
    }

@app.post("/unload/{model_id}")
def unload_model(model_id: str):
    """Manually unload a specific model to free memory"""
    if model_id not in MODELS:
        return JSONResponse(
            status_code=404,
            content={"error": f"Model {model_id} is not currently loaded"}
        )
    
    del MODELS[model_id]
    if model_id in MODEL_USAGE_TRACKER:
        del MODEL_USAGE_TRACKER[model_id]
    
    logger.info(f"🗑️ Manually unloaded model: {model_id}")
    return {
        "message": f"Model {model_id} unloaded successfully",
        "remaining_models": list(MODELS.keys()),
        "estimated_memory_freed_mb": MODEL_MEMORY_ESTIMATES.get(model_id, 100)
    }

@app.get("/favicon.ico")
def favicon():
    """Return empty favicon response so CRA dev-server proxy requests don't fail with ECONNREFUSED."""
    return Response(status_code=204)

@app.get("/models")
def list_models():
    """Return list of ALL available models - uses cached metadata for speed"""
    try:
        if DEFAULT_MODEL_ID and DEFAULT_MODEL_ID not in MODELS:
            load_single_model(DEFAULT_MODEL_ID)
        
        out = []
        for mid in AVAILABLE_MODEL_IDS:
            if mid in MODEL_METADATA_CACHE:
                out.append(MODEL_METADATA_CACHE[mid])
            else:
                out.append(get_model_metadata(mid))
        
        return {"models": out, "default": DEFAULT_MODEL_ID, "available": AVAILABLE_MODEL_IDS}
    except Exception as e:
        logger.error(f"Error in /models endpoint: {e}", exc_info=True)
        return {
            "models": [{"id": mid, "name": MODEL_DISPLAY_MAP.get(mid, mid)} for mid in AVAILABLE_MODEL_IDS],
            "default": DEFAULT_MODEL_ID,
            "available": AVAILABLE_MODEL_IDS,
            "error": str(e)
        }

@app.post("/predict")
async def predict(model_id: str = None, file: UploadFile = File(...), top_k: int = 4):
    try:
        content = await file.read()
        try:
            image = Image.open(io.BytesIO(content))
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            return JSONResponse(status_code=400, content={"error": "Invalid image file", "detail": str(e)})
        
        chosen = model_id or DEFAULT_MODEL_ID
        
        if chosen and chosen not in AVAILABLE_MODEL_IDS:
            return JSONResponse(
                status_code=400, 
                content={
                    "error": f"Model '{chosen}' not available", 
                    "available_models": AVAILABLE_MODEL_IDS
                }
            )
        
        if not chosen:
            return JSONResponse(status_code=400, content={"error": "No model_id provided and no default available"})

        if chosen not in MODELS:
            load_single_model(chosen)
        
        if chosen not in MODELS:
            return JSONResponse(status_code=500, content={"error": f"Failed to load model {chosen}"})

        model = MODELS[chosen]
        H, W = model.get("input_shape", (224, 224))
        input_name = model.get("input_name")
        session = model.get("session")

        arr = preprocess_image(image, target_size=(H or 224, W or 224))
        arr = arr.astype(np.float32)
        inputs = {input_name: arr}
        preds = session.run(None, inputs)[0]
        
        if preds.ndim == 2:
            probs = softmax(preds[0])
        elif preds.ndim == 1:
            probs = softmax(preds)
        else:
            probs = softmax(preds.ravel())

        top_idx = np.argsort(probs)[-top_k:][::-1]
        results = []
        labels = model.get("labels", [])
        for idx in top_idx:
            label = labels[idx] if labels and idx < len(labels) else str(int(idx))
            results.append({"label": label, "index": int(idx), "score": float(probs[idx])})

        return {"predictions": results}
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(e)})
