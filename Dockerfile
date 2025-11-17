# Stage 1: Construcción (build)
FROM python:3.11-slim as builder

WORKDIR /app

# Copiar requirements
COPY backend/requirements.txt .

# Instalar dependencias en una carpeta temporal
RUN pip install --user --no-cache-dir -r requirements.txt


# Stage 2: Producción (runtime)
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copiar las dependencias instaladas desde el builder
COPY --from=builder /root/.local /root/.local

# Configurar PATH para usar las dependencias
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production

# Crear carpeta de modelos primero
RUN mkdir -p /app/models

# Copiar el código de la aplicación
COPY backend/ ./backend/
COPY models/ ./models/

# Descargar modelos desde GitHub Releases si no existen
RUN python /app/backend/scripts/download_models.py

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Comando para iniciar
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
