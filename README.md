## Proyecto: servidor de inferencia + frontend React

Este repositorio contiene un backend en FastAPI que carga modelos ONNX y un frontend en React que permite subir imágenes y recibir predicciones. Está pensado para servir varios modelos (carpeta `models/` con subcarpetas por modelo).

Estructura principal
- `backend/` — código FastAPI, dependencias y README específico.
- `frontend/` — aplicación React (dev server con proxy a backend para desarrollo).
- `models/` — cada subcarpeta (`model_1`, `model_2`, ...) debe contener `model.onnx` y opcionalmente `config.json`/`config_base.json` y `metadata.json`.

Endpoints importantes (backend)
- `GET /models` — lista modelos cargados (id, name, input_size, labels, default).
- `GET /labels` — etiquetas del modelo por defecto.
- `POST /predict?model_id=<id>` — subir imagen (multipart form `file`) y obtener `predictions` (top-k). Si no se pasa `model_id` se usa el modelo por defecto.
- `GET /favicon.ico` — responde 204 para evitar mensajes de proxy en dev.

Requisitos
- Python 3.9+
- Node/npm (para el frontend)

Preparar y ejecutar (PowerShell)

1) Backend

```powershell
# Desde la raíz del repo
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
# Recomendado (desarrollo): escuchar en localhost
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

2) Frontend (desarrollo)

```powershell
cd frontend
npm install
npm start
```

Notas sobre CORS / proxy
- En desarrollo el `frontend/package.json` puede contener la clave `proxy` hacia `http://localhost:8000` para simplificar llamadas relativas. En producción sirve la build estática del frontend y configura el proxy/reverse-proxy (NGINX, CDN) según tu hosting.

Cómo organizar `models/`
- Cada modelo debe ir en su propia carpeta, por ejemplo:

```
models/
	model_1/
		model.onnx
		config.json         # opcional, con "classes" o "labels"
		metadata.json       # opcional, con "name" y descripción
	model_2/
		model.onnx
```

El backend escanea `models/` al iniciar y carga automáticamente cualquier `model.onnx` encontrado. Si quieres almacenar los pesos fuera del repo (recomendado para archivos grandes), sube los ONNX a S3/GCS/GitHub Release y añade un script de descarga en `backend/scripts/`.

Ejemplos de uso (curl)

```powershell
# Listar modelos
curl http://127.0.0.1:8000/models

# Predecir (archivo local)
curl -F "file=@C:\ruta\a\imagen.jpg" "http://127.0.0.1:8000/predict?model_id=model_1"
```

Respuesta `POST /predict` (ejemplo)

```json
{
	"model_id": "model_1",
	"predictions": [
		{"label":"Healthy","index":2,"score":0.92},
		{"label":"ESCA","index":1,"score":0.05}
	],
	"max_score": 0.92,
	"accepted": true
}
```

Buenas prácticas antes de subir a GitHub
- No comitees pesos grandes ni artefactos generados. Añade un `.gitignore` con reglas para excluir: `.venv/`, `*.pt`, `*.onnx` (si decidís no guardar ONNX en repo), carpetas `preds/`, `gradcam/`, `audit/`, CSVs de métricas, `frontend/node_modules/`, etc.
- Si ya comiste archivos grandes en el historial, usa `git rm --cached <archivo>` y commitea, o herramientas como `bfg` / `git filter-repo` para limpiar el historial.

Deploy — opciones rápidas
- Demo rápida: Frontend en Vercel y Backend en Render / Railway. Guarda modelos en S3/GCS y descárgalos al iniciar.
- Control con contenedores: crear `Dockerfile` para backend y publicar la imagen en Docker Hub / GHCR; desplegar en DigitalOcean App Platform, Render o AWS Cloud Run.

Debug y comprobaciones
- Si el frontend muestra `Proxy error: Could not proxy request ... (ECONNREFUSED)`, asegúrate de que el backend esté corriendo en `localhost:8000` o añade un `favicon.ico` en `frontend/public` para evitar peticiones proxied innecesarias.
- Revisar logs de uvicorn para errores de carga de ONNX.

Soporte y siguientes pasos
- Puedo:
	- Añadir `.gitignore` y limpiar el tracking Git de archivos grandes.
	- Añadir un script `backend/scripts/download_models.py` para descargar modelos desde URLs externas.
	- Dockerizar backend y añadir `docker-compose.yml` para testing local.

Si quieres que haga alguno de esos cambios (ej. crear `.gitignore` o Dockerfile), dime cuál y lo agrego.