# 🍇 Grape Disease Classifier - Production Stack

**Detector de enfermedades de plantas con IA Multi-Modelo usando FastAPI + React + ONNX**

![Status](https://img.shields.io/badge/Status-Production-green)
![Backend](https://img.shields.io/badge/Backend-FastAPI-blue)
![Frontend](https://img.shields.io/badge/Frontend-React%2018-lightblue)
![Inference](https://img.shields.io/badge/Inference-ONNX%20Runtime-orange)

---

## 📊 Overview

Sistema completo de clasificación de enfermedades en uvas con:

- **Backend**: FastAPI con lazy-loading de 4 modelos ONNX (~300MB total)
- **Frontend**: React SPA desplegada en Vercel
- **Infraestructura**: Docker containerizado, deployado en Render.com
- **Distribución de Modelos**: GitHub Releases + descarga automática en CI/CD

| Componente      | Tecnología          | Ambiente                                                     |
| --------------- | ------------------- | ------------------------------------------------------------ |
| **API Backend** | FastAPI 0.95.2      | Render (Python 3.11-slim)                                    |
| **Frontend**    | React 18.2          | Vercel                                                       |
| **Modelos**     | ONNX Runtime 1.20.1 | 4 arquitecturas (ConvNeXt, dlvtnet, mobilenetv3, swin_gsrdn) |
| **Database**    | N/A (stateless)     | -                                                            |

### ⚙️ Limitaciones del Plan Gratuito (Render Free Tier)

| Recurso         | Límite | Estado                       |
| --------------- | ------ | ---------------------------- |
| RAM             | 512 MB | ✅ Suficiente para 3 modelos |
| Modelos activos | 3/4    | ⚠️ model_4 deshabilitado     |
| Cold start      | 30s    | ✅ Aceptable                 |
| Almacenamiento  | 500 MB | ✅ Suficiente                |

**Upgrade a Pro ($12/mes):**

- 2 GB RAM (4 modelos simultáneos)
- Mejor performance
- Sin limitaciones de modelos

---

## 🏗️ Arquitectura & Estructura

```
grape-disease-classifier/
├── backend/                              # FastAPI server
│   ├── app.py                           # Main application (lazy-loading, CORS)
│   ├── requirements.txt                 # Python dependencies
│   ├── scripts/
│   │   └── download_models.py          # Auto-download from GitHub Releases
│   └── README.md                        # Backend-specific docs
│
├── frontend/                             # React SPA
│   ├── src/
│   │   ├── App.js                      # Multi-model selector UI
│   │   ├── index.js                    # React entry point
│   │   └── styles.css                  # Tailwind CSS
│   ├── package.json                    # Node dependencies
│   ├── public/index.html               # HTML template
│   └── .env.production                 # Production env vars
│
├── models/                               # Model directory (populated at runtime)
│   ├── model_1/                        # ConvNeXt
│   │   ├── model.onnx                  #
│   │   ├── config.json                 # Labels & metadata
│   │   └── metadata.json
│   ├── model_2/                        # dlvtnet
│   ├── model_3/                        # mobilenetv3
│   └── model_4/                        # swin_gsrdn
│
├── .github/workflows/                   # CI/CD (if using GH Actions)
├── docker-compose.yml                   # Local development
├── Dockerfile                           # Multi-stage production build
├── .dockerignore                        # Docker build exclusions
├── .gitignore                           # Git exclusions
├── .env.example                         # Environment template
└── README.md                            # This file

```

### Almacenamiento de Modelos

❌ **NO en Git** — Los archivos .onnx (~300MB) se excluyen de Git  
✅ **GitHub Releases** — Versioning de modelos + binarios

```
GitHub Release: v1.0.0
└── models.zip (294 MB)
    ├── models/model_1/model.onnx
    ├── models/model_2/model.onnx
    ├── models/model_3/model.onnx
    └── models/model_4/model.onnx
```

---

## 🚀 Deployment Architecture

### Production Stack

```
┌─────────────────────────────────────────────────┐
│          Vercel (Frontend)                      │
│  https://grape-disease-classifier.vercel.app  │
│  - React SPA (auto-deploy from GitHub)         │
│  - REACT_APP_API_URL env var                   │
└──────────────────┬──────────────────────────────┘
                   │ CORS HTTP Requests
                   ↓
┌─────────────────────────────────────────────────┐
│         Render (Backend)                        │
│  https://grape-disease-classifier-backend...  │
│  - FastAPI container (512MB RAM free tier)     │
│  - Auto-deploy from GitHub                     │
│  - GitHub Release model download               │
│  - Health checks + metrics                     │
└────────────────────────────────────────────────┘
              │
              ├─→ Lazy-load models on-demand
              ├─→ Metadata cache (startup)
              └─→ Exception handling (502 prevention)

┌─────────────────────────────────────────────────┐
│      GitHub Releases (Model Distribution)      │
│  - Version: v1.0.0+                            │
│  - models.zip (~294 MB)                        │
│  - Auto-download in Dockerfile                 │
└─────────────────────────────────────────────────┘
```

---

## 🔧 Stack Tecnológico

### Backend

```
Python 3.11-slim (Docker)
├── fastapi==0.95.2              # Web framework
├── uvicorn==0.22.0              # ASGI server
├── onnxruntime==1.20.1          # Inference engine
├── pillow==11.0.0               # Image processing
└── numpy==2.3.0                 # Array operations
```

### Frontend

```
Node 18-alpine (Docker) → Vercel
├── react==18.2.0                # UI framework
├── tailwindcss                  # Styling
├── lucide-react                 # Icons
└── vite (optional)              # Build tool
```

### DevOps

- **Containerization**: Docker (multi-stage builds)
- **Orchestration**: Docker Compose (local)
- **CI/CD**: GitHub Actions (optional)
- **Version Control**: Git + GitHub
- **Model Distribution**: GitHub Releases
- **Hosting**: Render + Vercel

---

## 📥 Quick Start

### Desarrollo Local

#### 1. Clonar y preparar entorno

```bash
git clone https://github.com/oscarperdomop/grape-disease-classifier.git
cd grape-disease-classifier
```

#### 2. Backend (Python 3.11+)

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r backend/requirements.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Backend disponible en: `http://localhost:8000`

#### 3. Frontend (Node 18+)

```bash
cd frontend
npm install
npm start
```

Frontend disponible en: `http://localhost:3000`

#### 4. Descargar modelos (si no existen)

```bash
python backend/scripts/download_models.py
```

### Docker (Local)

```bash
docker-compose up --build
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

---

## 🌐 API Reference

### Base URL

- **Local**: `http://localhost:8000`
- **Production**: `https://grape-disease-classifier-backend.onrender.com`

### Endpoints

#### `GET /models`

Lista todos los modelos disponibles con metadatos.

**Response:**

```json
{
  "models": [
    {
      "id": "model_1",
      "name": "ConvNeXt",
      "framework": "onnx",
      "num_labels": 4,
      "labels": ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]
    }
  ],
  "default": "model_1",
  "available": ["model_1", "model_2", "model_3", "model_4"]
}
```

**Performance**: <100ms (cached at startup)

---

#### `POST /predict`

Realizar predicción sobre una imagen.

**Parameters:**

- `model_id` (query, optional): ID del modelo (default: model_1)
- `file` (form, required): Archivo de imagen (JPG/PNG)
- `top_k` (query, optional): Top K resultados (default: 4)

**Request:**

```bash
curl -X POST "https://api.example.com/predict?model_id=model_1&top_k=5" \
  -F "file=@leaf.jpg"
```

**Response:**

```json
{
  "predictions": [
    {
      "label": "Healthy",
      "index": 2,
      "score": 0.923
    },
    {
      "label": "ESCA",
      "index": 1,
      "score": 0.062
    }
  ]
}
```

**Performance (First): 5-10s (model load)** | **Subsequent: <2s**

---

#### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "ok",
  "models_available": 4,
  "models_loaded": 1,
  "default_model": "model_1"
}
```

---

#### `GET /debug`

Información de debug (modelo development).

**Response:**

```json
{
  "models_dir": "/app/models",
  "models_dir_exists": true,
  "models_dir_contents": ["model_1", "model_2", "model_3", "model_4"],
  "available_models": ["model_1", "model_2", "model_3", "model_4"],
  "models_currently_loaded": 1,
  "default_model": "model_1"
}
```

---

## ⚙️ Configuración de Entorno

### Variables de Entorno

**Backend** (`.env` en root o Render):

```bash
# Ya configurados en código
MODELS_DIR=/app/models
CORS_ORIGINS=["*"]
LOG_LEVEL=INFO
```

**Frontend** (Vercel):

```bash
REACT_APP_API_URL=https://grape-disease-classifier-backend.onrender.com
```

---

## 🚢 Deployment

### Deploy a Producción

#### Opción A: Render + Vercel (Recomendado)

**Backend (Render):**

1. Conectar repo GitHub a Render
2. Create Web Service
3. Build command: `pip install -r backend/requirements.txt`
4. Start command: `uvicorn backend.app:app --host 0.0.0.0 --port 8000`
5. Environment: `PYTHONUNBUFFERED=true`

**Frontend (Vercel):**

1. Conectar repo GitHub a Vercel
2. Framework: React
3. Environment: `REACT_APP_API_URL=https://your-render-backend.onrender.com`
4. Auto-deploy on push

#### Opción B: Docker (Self-hosted)

```bash
docker build -t grape-disease-classifier .
docker run -p 8000:8000 -e PYTHONUNBUFFERED=true grape-disease-classifier
```

---

## 🔄 Model Updates

### Crear Nueva Release con Modelos

```bash
# 1. Crear ZIP con modelos
Compress-Archive -Path models/ -DestinationPath models.zip

# 2. Crear GitHub Release
# Ir a: https://github.com/oscarperdomop/grape-disease-classifier/releases
# - Tag: v1.0.1
# - Upload: models.zip
# - Publish

# 3. Deploy automático en Render
# (webhook automático descarga nueva versión)
```

---

## 📊 Performance & Optimizations

### Lazy Loading Architecture

✅ **Startup**: Metadata cacheado, 0 modelos en memoria (~50MB)
✅ **Primera predicción**: Cargar modelo on-demand (~5-10s)
✅ **Predicciones siguientes**: Modelo en caché (~<2s)
✅ **Escalabilidad**: Render 512MB free tier suficiente

### Caching Strategy

```python
# Startup (discover_models)
AVAILABLE_MODEL_IDS = [...]              # Scan folder
MODEL_METADATA_CACHE = {...}             # Load config.json

# Runtime (/models endpoint)
return cached_metadata                   # <100ms response

# On-demand (/predict endpoint)
if model not in MODELS:
    load_single_model(model_id)         # First request: 5-10s
return predictions                       # Cached: <2s
```

### Memory Management

| Fase                   | Memoria | Modelos Cargados | Plan             |
| ---------------------- | ------- | ---------------- | ---------------- |
| Startup                | ~50MB   | 0                | Free             |
| /models request        | ~50MB   | 0                | Free             |
| 1er /predict (model_1) | ~120MB  | 1 (ConvNeXt)     | Free             |
| 2do /predict (model_2) | ~180MB  | 2 (dlvtnet)      | Free             |
| 3er /predict (model_3) | ~250MB  | 3 (mobilenetv3)  | Free             |
| +/predict (model_4)    | >400MB  | 4 (swin_gsrdn)   | ❌ Free / ✅ Pro |

**Render free tier: 512MB** → **3 modelos máximo** ⚠️

**Solución**:

- En producción, model_4 se desactiva automáticamente
- El frontend muestra aviso: "Modelos deshabilitados en plan gratuito"
- Upgrade a Pro ($12/mes) para usar todos los 4 modelos

---

## 🐛 Troubleshooting

### 502 Bad Gateway - model_4 (swin_gsrdn) crashes

**Causa**: model_4 es muy pesado (~150MB+) y excede memoria del plan free (512MB)

**Síntoma**: Funciona con model_1/2/3 pero falla con model_4

**Solución**:

```bash
# En DESARROLLO: Funciona todo localmente
docker-compose up

# En PRODUCCIÓN (Render free tier):
# ✅ model_1, model_2, model_3 disponibles
# ❌ model_4 automáticamente deshabilitado
# Frontend muestra: "Modelos deshabilitados en plan gratuito"

# Upgrade a Pro:
# 1. Render Dashboard → Service settings → Change plan
# 2. Select "Pro" ($12/month, 2GB RAM)
# 3. Auto-redeploy con todos los 4 modelos
```

---

### Modelos no cargan

**Causa**: Archivo de configuración corrompido o encoding inválido

**Solución**:

```bash
# Verificar archivos config
cat models/model_1/config.json | jq .

# Rebuild backend
docker-compose up --build
```

### Modelos no cargan

**Causa**: Ruta incorrecta o archivo .onnx corrupto

**Solución**:

```bash
# Check /debug endpoint
curl http://localhost:8000/debug

# Verificar ZIP de GitHub Release
# Descargar y verificar estructura
```

### CORS Error en Frontend

**Causa**: Backend no retorna `Access-Control-Allow-Origin`

**Verificar**:

```python
# backend/app.py debe tener:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specific origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📈 Monitoring & Metrics

### Health Checks

**Render**: Auto-configured en `/health` endpoint

```bash
curl https://grape-disease-classifier-backend.onrender.com/health
```

### Logs

**Render**:

```bash
# CLI
render logs --tail 100

# Dashboard
https://dashboard.render.com → Service logs
```

**Local**:

```bash
# Backend
docker logs -f grape-disease-classifier-backend

# Frontend (Vercel)
Vercel Dashboard → Deployments → Logs
```

---

## 🔐 Security Best Practices

- ✅ `.gitignore` excluye modelos + secrets
- ✅ CORS configurado apropiadamente (production: specific origins)
- ✅ Input validation en `/predict` (file type check)
- ✅ Exception handling previene información sensitiva en errors
- ✅ No secrets en código (env vars)
- ✅ HTTPS en producción (automático en Render + Vercel)

### Recomendaciones Futuras

```python
# 1. Rate limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

# 2. Authentication (si necesario)
from fastapi.security import HTTPBearer

# 3. Logging & Monitoring
from pythonjsonlogger import jsonlogger
```

---

## 📚 Documentación Adicional

- [FastAPI Docs](http://localhost:8000/docs) — Swagger UI (development)
- [API Schema](http://localhost:8000/openapi.json) — OpenAPI spec
- [Backend README](./backend/README.md) — Backend-specific setup
- [Render Docs](https://render.com/docs) — Deployment guide
- [Vercel Docs](https://vercel.com/docs) — Frontend deployment

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m 'Add my feature'`
4. Push to branch: `git push origin feature/my-feature`
5. Open Pull Request

---

## 📄 License

MIT License - See LICENSE file

---

## 👨‍💻 Author

**Oscar Perdomo** - GitHub: [@oscarperdomop](https://github.com/oscarperdomop)

---

## 🙏 Acknowledgments

- ONNX Runtime for efficient inference
- FastAPI for simple async APIs
- React for reactive UI
- Render & Vercel for serverless hosting

---

**Last Updated**: November 2025 | **Status**: Production Ready ✅
