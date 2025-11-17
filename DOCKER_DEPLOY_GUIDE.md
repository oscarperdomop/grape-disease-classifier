# 📚 Guía Completa: Docker → Render → Vercel (Sin experiencia previa)

## ¿Qué es Docker?

Docker es como un "contenedor" que empaqueta tu aplicación con todo lo que necesita (Python, librerías, modelos ONNX) en una caja. Cuando la subes a Render, la caja funciona exactamente igual que en tu computadora.

**Analogía:** Es como meter tu proyecto en una caja sellada, y esa caja funciona igual en cualquier parte.

---

## PASO 1: Instalar Docker (2 minutos)

### En Windows:

1. Descarga **Docker Desktop** desde: https://www.docker.com/products/docker-desktop
2. Instala normalmente (next, next, finish)
3. Reinicia la computadora
4. Abre PowerShell y verifica:

```powershell
docker --version
```

Deberías ver algo como: `Docker version 27.0.1`

---

## PASO 2: Probar Docker localmente (5 minutos)

Esto construye la caja y la prueba en tu compu antes de subirla a Render.

### En PowerShell (en la raíz del proyecto):

```powershell
# Construir la imagen Docker
docker build -t grape-disease-classifier .

# Ejecutar el contenedor
docker run -p 8000:8000 grape-disease-classifier
```

**¿Qué está pasando?**

- `docker build` — crea la caja con tu código + dependencias
- `docker run` — inicia la caja y expone el puerto 8000

Si ves algo como:

```
Uvicorn running on http://0.0.0.0:8000
```

✅ **¡Funciona!** Presiona `Ctrl+C` para detener.

---

## PASO 3: Subir código a GitHub (5 minutos)

Docker necesita tu código en GitHub para que Render lo descargue.

### Si NO tienes Git configurado:

```powershell
# Configurar Git (primera vez)
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"

# Crear repo local
git init
```

### Subir el código:

```powershell
# Agregar todos los archivos
git add .

# Confirmar cambios
git commit -m "Add Docker support and production config"

# Crear repositorio en GitHub (si no existe)
# 1. Ve a https://github.com/new
# 2. Crea repo "grape-disease-classifier"
# 3. Copia la URL (ej: https://github.com/tuusuario/grape-disease-classifier.git)

# Conectar con GitHub (reemplaza la URL)
git remote add origin https://github.com/TU_USUARIO/grape-disease-classifier.git
git branch -M main
git push -u origin main
```

**Nota:** Si pide contraseña, usa token personal: https://github.com/settings/tokens (crea uno con permisos `repo`)

---

## PASO 4: Deploy Backend en Render (10 minutos)

Render construirá la imagen Docker y la ejecutará en la nube.

### 1. Crear cuenta en Render

- Ve a https://render.com
- Registrate con GitHub (más fácil)

### 2. Crear servicio

1. Click en **"New +"** → **"Web Service"**
2. Conecta tu repositorio de GitHub (autoriza Render)
3. Selecciona: `grape-disease-classifier`
4. Llena los datos:
   - **Name:** `grape-disease-classifier-backend`
   - **Environment:** `Docker` (automático)
   - **Branch:** `main`
   - **Build Command:** (dejar vacío, usa Dockerfile)
   - **Start Command:** (dejar vacío, usa Dockerfile)
5. **Plan:** Free (gratis con limitaciones)
6. Click **"Create Web Service"**

### 3. Esperar deployment

- Render construirá la imagen (2-3 minutos)
- Verás logs: `Building image...` → `Starting service...`
- Cuando veas ✅, el backend está corriendo

### 4. Obtener URL del backend

En Render, copia la URL (ej: `https://grape-disease-classifier.onrender.com`)

---

## PASO 5: Deploy Frontend en Vercel (5 minutos)

Vercel automáticamente detecta React y lo deploya.

### 1. Ir a Vercel

- Ve a https://vercel.com
- Click **"Continuar con GitHub"**
- Autoriza y conecta

### 2. Importar proyecto

1. Click **"Add New..."** → **"Project"**
2. Selecciona: `grape-disease-classifier`
3. Configura:
   - **Framework:** React
   - **Root Directory:** `frontend` ← IMPORTANTE

### 3. Variables de entorno

Antes de deployar, agregar variable:

```
REACT_APP_API_URL = https://grape-disease-classifier.onrender.com
```

(Reemplaza con tu URL de Render)

### 4. Deploy

Click **"Deploy"** y espera (1-2 minutos)

Cuando veas ✅, tu frontend está en línea.

---

## PASO 6: Conectar Frontend ↔ Backend (2 minutos)

Necesitas actualizar el frontend para usar la URL de Render en producción.

### En `frontend/src/App.js` (o donde hagas llamadas):

```javascript
// Detectar si estamos en desarrollo o producción
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Usar en tus fetch() calls:
fetch(`${API_URL}/predict`, ...)
```

### Commit y push:

```powershell
git add .
git commit -m "Add production API URL"
git push origin main
```

Vercel automáticamente re-deployará el frontend.

---

## PASO 7: Prueba final

1. Abre el frontend en Vercel (la URL que te da)
2. Sube una imagen
3. Verifica que aparezcan predicciones
4. ✅ ¡Listo!

---

## Troubleshooting (Si algo falla)

### "Docker command not found"

→ Instalar Docker Desktop y reiniciar

### "Render dice: image build failed"

→ Ver logs en Render:

1. Click en el servicio
2. Ir a **Logs** → **Build Logs**
3. Buscar el error (generalmente es algo en requirements.txt)

### "Frontend no conecta con backend"

→ Verificar:

1. URL de Render es correcta en `REACT_APP_API_URL`
2. Backend está corriendo (ver Render logs)
3. CORS está habilitado en `backend/app.py`

### "Modelos muy pesados ralentizan startup"

→ Solución (próximo paso):

- Subir modelos a GitHub Releases
- Descargar automáticamente al iniciar
- (Puedo implementar esto si necesitas)

---

## Próximos pasos (Opcional)

- ✅ Agregar autoscaling en Render (plan pago)
- ✅ Cachear modelos para arranque más rápido
- ✅ Agregar CI/CD automático (GitHub Actions)
- ✅ Monitoreo y logs (Sentry)

---

## Comandos útiles Docker (Referencia)

```powershell
# Ver imágenes construidas
docker images

# Ver contenedores corriendo
docker ps

# Detener un contenedor
docker stop <CONTAINER_ID>

# Ver logs de un contenedor
docker logs <CONTAINER_ID>

# Eliminar imagen
docker rmi <IMAGE_ID>
```

---

**¿Preguntas? Avísame en qué paso te quedaste y ayudo.**
