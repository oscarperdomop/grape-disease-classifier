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

# Enable CORS for frontend
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
VALIDATOR_MODEL = None  # Validation model to check if image is grape leaf

# Human-friendly names for known folders
MODEL_DISPLAY_MAP = {
    "model_1": "ConvNeXt",
    "model_2": "dlvtnet",
    "model_3": "mobilenetv3",
    "model_4": "swin_gsrdn",
}

# Validation model configuration
VALIDATOR_MODEL_ID = "validator"  # Special model for image validation
VALIDATOR_MODEL_THRESHOLD = 0.5  # Confidence threshold for accepting as grape leaf
# NOTE: Validator model is OPTIONAL. If not present, all images are accepted (fail-safe mode)
# To enable validation, train a model and place it in models/validator/model.onnx
# See VALIDATOR_MODEL_GUIDE.md and train_validator_simple.py for instructions

# Production environment flag
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"

# Memory management configuration - AGGRESSIVE CLEANUP STRATEGY
# After each prediction, unload ALL models to give full memory to next request
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "450"))  # More memory available since we cleanup after each request
MODEL_MEMORY_ESTIMATES = {
    "model_1": 80,
    "model_2": 70,
    "model_3": 60,
    "model_4": 200,  # model_4 can use more memory since it's alone
}
MAX_CONCURRENT_MODELS = int(os.getenv("MAX_CONCURRENT_MODELS", "1"))  # Only 1 model at a time
MODEL_USAGE_TRACKER = {}
AGGRESSIVE_CLEANUP = True  # Unload all models after each prediction

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
        # Fallback if psutil not available
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
    
    # Check if adding this model would exceed limits
    if current_memory + new_model_memory > MAX_MEMORY_MB:
        return False
    
    # Check concurrent model limit
    if len(MODELS) >= MAX_CONCURRENT_MODELS and model_id not in MODELS:
        return False
        
    return True

def unload_least_recently_used_model():
    """Unload the least recently used model to free memory"""
    if not MODELS or not MODEL_USAGE_TRACKER:
        return None
        
    # Find least recently used model
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

def load_validator_model():
    """Load the validation model to check if image is a grape leaf"""
    global VALIDATOR_MODEL
    
    if VALIDATOR_MODEL is not None:
        return VALIDATOR_MODEL
    
    validator_folder = os.path.join(MODELS_DIR, VALIDATOR_MODEL_ID)
    validator_path = os.path.join(validator_folder, "model.onnx")
    
    if not os.path.exists(validator_path):
        logger.warning(f"⚠️  Validator model not found at {validator_path}")
        return None
    
    try:
        logger.info(f"🔍 Loading validator model...")
        sess = ort.InferenceSession(validator_path, providers=["CPUExecutionProvider"])
        meta = sess.get_inputs()[0]
        shape = meta.shape
        H = safe_int(shape[2]) or 224
        W = safe_int(shape[3]) or 224
        
        VALIDATOR_MODEL = {
            "session": sess,
            "input_name": meta.name,
            "input_shape": (H, W),
        }
        logger.info(f"✅ Validator model loaded successfully")
        return VALIDATOR_MODEL
    except Exception as e:
        logger.error(f"❌ Failed to load validator model: {e}")
        return None

def validate_grape_leaf(image: Image.Image) -> tuple[bool, float]:
    """
    Validate if image is a grape leaf using validator model
    Returns: (is_valid, confidence)
    """
    if VALIDATOR_MODEL is None:
        logger.warning("⚠️  Validator model not loaded - validation disabled")
        logger.warning("   To enable validation, add validator model to models/validator/model.onnx")
        return True, 1.0  # If no validator, accept image (fail-safe)
    
    try:
        H, W = VALIDATOR_MODEL.get("input_shape", (224, 224))
        input_name = VALIDATOR_MODEL.get("input_name")
        session = VALIDATOR_MODEL.get("session")
        
        arr = preprocess_image(image, target_size=(H or 224, W or 224))
        arr = arr.astype(np.float32)
        inputs = {input_name: arr}
        preds = session.run(None, inputs)[0]
        
        # Assuming binary classification: [not_grape, is_grape]
        if preds.ndim == 2:
            probs = softmax(preds[0])
        elif preds.ndim == 1:
            probs = softmax(preds)
        else:
            probs = softmax(preds.ravel())
        
        # Get confidence for "is grape leaf" class (usually index 1)
        grape_confidence = float(probs[-1])  # Last class is typically positive
        is_valid = grape_confidence >= VALIDATOR_MODEL_THRESHOLD
        
        logger.info(f"🔍 Validation result: {'✅ Valid' if is_valid else '❌ Invalid'} (confidence: {grape_confidence:.2%})")
        return is_valid, grape_confidence
    
    except Exception as e:
        logger.error(f"❌ Validation error: {e}")
        return True, 1.0  # On error, accept image (fail-safe)

def cleanup_all_models():
    """Aggressively unload ALL models to free memory for next request"""
    global MODELS, MODEL_USAGE_TRACKER, VALIDATOR_MODEL
    if MODELS:
        logger.info(f"🧹 Aggressive cleanup: Unloading {len(MODELS)} models")
        for model_id in list(MODELS.keys()):
            logger.info(f"   🗑️  Unloading {model_id}")
            del MODELS[model_id]
        MODEL_USAGE_TRACKER.clear()
    
    # Keep validator model loaded (it's small and needed for every request)
    # Only unload other models
    
    # Force garbage collection
    import gc
    gc.collect()
    logger.info(f"✅ Memory cleanup complete. Available for next request.")

def get_model_metadata(model_id: str):
    """Get metadata for a model (name, labels) without loading ONNX"""
    display_name = MODEL_DISPLAY_MAP.get(model_id, model_id)
    labels = []
    
    model_folder = os.path.join(MODELS_DIR, model_id)
    
    # Try to load labels from config files
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
    """Scan MODELS_DIR and list available models WITHOUT loading them (lazy loading)"""
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
            # All models are now available with smart memory management
            AVAILABLE_MODEL_IDS.append(entry)
            # Pre-load metadata at startup to avoid slow /models requests
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
    
    # Special handling for model_4: it's too heavy, unload everything first
    if model_id == "model_4" and len(MODELS) > 0:
        logger.info(f"⚠️  model_4 requires exclusive memory - unloading all other models")
        for mid in list(MODELS.keys()):
            logger.info(f"🗑️  Unloading {mid} to make room for model_4")
            del MODELS[mid]
            if mid in MODEL_USAGE_TRACKER:
                del MODEL_USAGE_TRACKER[mid]
    
    # Check if we can load this model
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
        
        # Try to load with timeout protection
        try:
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        except MemoryError as me:
            logger.error(f"❌ MemoryError loading {model_id}: {me}")
            logger.error("Model is too large for available memory. Consider upgrading plan.")
            return None
        except Exception as load_err:
            logger.error(f"❌ Error loading {model_id}: {load_err}")
            return None
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
        display_name = model_id
        metadata_path = os.path.join(model_folder, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    md = json.load(f)
                    display_name = md.get("name") or display_name
            except Exception:
                pass
        
        # override with our friendly names when available
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
        
        # Update usage tracker
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
    # Load validator model at startup (it stays in memory)
    load_validator_model()

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


@app.get("/validator")
def validator_info():
    """Get information about the validator model"""
    return {
        "validator_enabled": VALIDATOR_MODEL is not None,
        "validator_model_id": VALIDATOR_MODEL_ID,
        "validator_threshold": VALIDATOR_MODEL_THRESHOLD,
        "description": "Binary classifier that validates if image is a grape leaf",
        "usage": "Automatically runs before disease classification",
        "response_on_invalid": {
            "error": "Image is not in scope",
            "message": "The image does not appear to be a grape leaf"
        }
    }

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
        "validator_loaded": VALIDATOR_MODEL is not None,
        "can_load_more": len(MODELS) < MAX_CONCURRENT_MODELS,
        "model_memory_estimates": MODEL_MEMORY_ESTIMATES,
        "aggressive_cleanup_enabled": AGGRESSIVE_CLEANUP,
        "strategy": "Unload ALL models after each prediction to maximize memory for next request"
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
        # Load default model if not already loaded to populate labels
        if DEFAULT_MODEL_ID and DEFAULT_MODEL_ID not in MODELS:
            load_single_model(DEFAULT_MODEL_ID)
        
        # Return all models using cached metadata
        out = []
        for mid in AVAILABLE_MODEL_IDS:
            if mid in MODEL_METADATA_CACHE:
                out.append(MODEL_METADATA_CACHE[mid])
            else:
                # Fallback (shouldn't happen if discover_models ran)
                out.append(get_model_metadata(mid))
        
        return {"models": out, "default": DEFAULT_MODEL_ID, "available": AVAILABLE_MODEL_IDS}
    except Exception as e:
        logger.error(f"Error in /models endpoint: {e}", exc_info=True)
        # Return at least the available model IDs even if metadata fails
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
        
        # choose model
        chosen = model_id or DEFAULT_MODEL_ID
        
        # Check if model exists in filesystem
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

        # STEP 1: Validate if image is a grape leaf
        is_valid, validation_confidence = validate_grape_leaf(image)
        if not is_valid:
            logger.warning(f"⚠️  Image rejected by validator (confidence: {validation_confidence:.2%})")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Image is not in scope",
                    "message": "The image does not appear to be a grape leaf",
                    "validation_confidence": round(validation_confidence, 4),
                    "note": "Please provide an image of a grape leaf for analysis"
                }
            )

        # STEP 2: Load model on-demand if not already loaded
        if chosen not in MODELS:
            load_single_model(chosen)
        
        if chosen not in MODELS:
            return JSONResponse(status_code=500, content={"error": f"Failed to load model {chosen}"})

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

        # AGGRESSIVE CLEANUP: Unload all models after prediction to free memory for next request
        if AGGRESSIVE_CLEANUP:
            cleanup_all_models()
        
        return {"predictions": results}
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        
        # Still cleanup on error
        if AGGRESSIVE_CLEANUP:
            cleanup_all_models()
        
        return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(e)})
