# 🔑 GitHub Token Setup para Render

## Problema
El deployment falla con error `HTTP Error 403: rate limit exceeded` porque GitHub API tiene límite de 60 requests/hora sin autenticación.

## Solución
Usar un **GitHub Personal Access Token** para aumentar el límite a 5000 requests/hora.

---

## 📋 Pasos para Configurar

### 1. Crear un GitHub Personal Access Token

1. Ve a: https://github.com/settings/tokens
2. Click en **"Generate new token"** → **"Generate new token (classic)"**
3. Llena los campos:
   - **Token name**: `grape-disease-classifier-render`
   - **Expiration**: `90 days` (o más)
   - **Scopes**: Selecciona solo `public_repo` (acceso a repos públicos)
4. Click en **"Generate token"**
5. **Copia el token** (aparece una sola vez)

### 2. Agregar Token a Render

1. Ve a tu servicio en Render Dashboard: https://dashboard.render.com
2. Selecciona `grape-disease-classifier-backend`
3. Ve a **Settings** → **Environment**
4. Click en **"Add Environment Variable"**
5. Agrega:
   - **Key**: `GITHUB_TOKEN`
   - **Value**: `<pega el token aquí>`
6. Click en **"Save Changes"**

### 3. Triggear un Nuevo Deployment

1. En Render Dashboard, ve a tu servicio
2. Click en **"Manual Deploy"** o **"Redeploy latest commit"**
3. Espera a que termine el build (2-3 minutos)

---

## ✅ Verificación

Después del deployment, verifica en los logs:

```
INFO:__main__:Using GitHub token for authentication
INFO:__main__:Downloading models from: https://github.com/oscarperdomop/grape-disease-classifier/releases/download/...
```

Si ves esto, ¡funcionó! Los modelos se descargaron correctamente.

---

## 🔒 Seguridad

- El token solo tiene acceso a repos **públicos** (`public_repo`)
- No puede modificar nada, solo leer
- Expira en 90 días (configurable)
- Puedes revocar el token en cualquier momento en GitHub Settings

---

## 🆘 Troubleshooting

**Si aún falla:**

1. Verifica que el token esté correctamente copiado (sin espacios)
2. Asegúrate de que el repo tenga una Release con modelos
3. Revisa los logs en Render para más detalles
4. Intenta crear un nuevo token

**Si necesitas deshabilitar el token:**

1. Ve a https://github.com/settings/tokens
2. Click en el token
3. Click en **"Delete"**
4. Render seguirá funcionando pero sin modelos (fallará en predicciones)
