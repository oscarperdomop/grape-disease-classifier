# Backend (FastAPI) de inferencia

Requisitos: Python 3.9+.

Instalación (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

Ejecutar el servidor (PowerShell):

```powershell
# Desde la raíz del repo
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

El endpoint principal es `POST /predict` y acepta un archivo de imagen `file` (multipart/form-data) y un parámetro opcional `top_k`.

Nuevos endpoints para múltiples modelos:
- `GET /models` — devuelve la lista de modelos disponibles encontrados en la carpeta `models/` (id, name, labels, default).
- `POST /predict?model_id=<model_id>` — especifica `model_id` (nombre de la subcarpeta dentro de `models/`) para usar un modelo concreto. Si no se pasa `model_id`, se usará el modelo por defecto (primero cargado).

El backend busca carpetas en `models/` que contengan `model.onnx` y carga automáticamente los modelos ONNX.