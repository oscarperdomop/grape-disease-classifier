#!/usr/bin/env python3
"""
Script para descargar modelos ONNX desde GitHub Releases.
Se ejecuta automáticamente durante el build de Docker.
"""

import os
import sys
import json
import urllib.request
import urllib.error
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
GITHUB_OWNER = "oscarperdomop"
GITHUB_REPO = "grape-disease-classifier"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"

# Obtener ruta del directorio de modelos
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

def get_download_url():
    """Obtiene la URL de descarga desde la última Release de GitHub"""
    try:
        logger.info(f"Fetching latest release info from {GITHUB_API_URL}")
        with urllib.request.urlopen(GITHUB_API_URL, timeout=10) as response:
            data = json.loads(response.read().decode())
            
        if "assets" not in data or len(data["assets"]) == 0:
            logger.warning("No assets found in latest release")
            return None
        
        # Buscar el archivo .zip con los modelos
        for asset in data["assets"]:
            if asset["name"].endswith(".zip") and "models" in asset["name"].lower():
                return asset["browser_download_url"]
        
        logger.warning("No models zip file found in release assets")
        return None
    
    except urllib.error.URLError as e:
        logger.error(f"Failed to fetch release info: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

def download_models(download_url):
    """Descarga y extrae los modelos"""
    try:
        import zipfile
        import tempfile
        import shutil
        
        logger.info(f"Downloading models from: {download_url}")
        
        # Descargar a un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            tmp_path = tmp_file.name
            
        # Descargar
        urllib.request.urlretrieve(download_url, tmp_path)
        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        logger.info(f"Downloaded to {tmp_path} ({file_size_mb:.1f} MB)")
        
        # Crear directorio de extracción temporal
        extract_temp = Path(tempfile.gettempdir()) / "models_extract"
        if extract_temp.exists():
            shutil.rmtree(extract_temp)
        extract_temp.mkdir(parents=True)
        
        # Extraer TODO
        logger.info(f"Extracting ZIP to temporary directory: {extract_temp}")
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            zip_ref.extractall(extract_temp)
        
        # Listar contenido extraído
        logger.info(f"Contents extracted:")
        for item in sorted(extract_temp.rglob("*")):
            if item.is_file():
                logger.info(f"  FILE: {item.relative_to(extract_temp)}")
            else:
                logger.info(f"  DIR:  {item.relative_to(extract_temp)}")
        
        # Buscar carpeta "models"
        models_source = extract_temp / "models"
        
        if not models_source.exists():
            logger.error(f"Models folder not found at {models_source}")
            logger.error("Available directories:")
            for item in extract_temp.iterdir():
                logger.error(f"  {item.name}")
            return False
        
        # Copiar todos los archivos de modelos
        logger.info(f"Copying models from {models_source} to {MODELS_DIR}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Limpiar MODELS_DIR primero
        for item in MODELS_DIR.iterdir():
            if item.name != ".gitkeep":
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        
        # Copiar carpetas de modelos
        copy_count = 0
        for model_folder in models_source.iterdir():
            if model_folder.is_dir():
                dest = MODELS_DIR / model_folder.name
                shutil.copytree(model_folder, dest)
                logger.info(f"  ✓ Copied {model_folder.name}")
                copy_count += 1
        
        if copy_count == 0:
            logger.error("No model folders found to copy")
            return False
        
        # Limpiar temporales
        os.remove(tmp_path)
        shutil.rmtree(extract_temp)
        logger.info(f"✓ Models downloaded and extracted successfully ({copy_count} folders)")
        return True
    
    except Exception as e:
        logger.error(f"Failed to download/extract models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def check_models():
    """Verifica que los modelos se hayan descargado correctamente"""
    model_count = 0
    
    if not MODELS_DIR.exists():
        logger.warning(f"Models directory does not exist: {MODELS_DIR}")
        return 0
    
    for model_folder in MODELS_DIR.iterdir():
        if model_folder.is_dir() and not model_folder.name.startswith('.'):
            onnx_file = model_folder / "model.onnx"
            if onnx_file.exists():
                model_count += 1
                logger.info(f"✓ Found model: {model_folder.name}")
            else:
                logger.warning(f"✗ No model.onnx in {model_folder.name}")
    
    logger.info(f"Total models found: {model_count}")
    return model_count

def main():
    """Función principal"""
    logger.info("=" * 60)
    logger.info("Model Download Script")
    logger.info("=" * 60)
    
    # Verificar si ya existen modelos
    existing_models = check_models()
    if existing_models > 0:
        logger.info(f"Models already present ({existing_models} found). Skipping download.")
        return 0
    
    # Obtener URL de descarga
    download_url = get_download_url()
    if not download_url:
        logger.warning("Could not get download URL. Models may not be available.")
        logger.warning("Ensure you have created a GitHub Release with model files.")
        logger.info("Continuing without models (API will start but predictions may fail)")
        return 0  # No es error crítico
    
    # Descargar
    if not download_models(download_url):
        logger.error("Failed to download models")
        logger.warning("Continuing without models (API will start but predictions may fail)")
        return 0  # No es error crítico
    
    # Verificar descarga
    final_count = check_models()
    if final_count == 0:
        logger.error("No models found after download")
        logger.warning("Continuing without models (API will start but predictions may fail)")
        return 0  # No es error crítico
    
    logger.info("=" * 60)
    logger.info("✓ Script completed successfully")
    logger.info("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
