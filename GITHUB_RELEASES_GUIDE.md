# 📦 Cómo crear una Release de GitHub con los modelos

## Paso 1: Preparar los archivos localmente

En tu computadora, en la carpeta raíz del proyecto:

```bash
# Crear un ZIP con todos los modelos
# Windows (PowerShell)
Compress-Archive -Path models/ -DestinationPath models.zip

# O macOS/Linux
zip -r models.zip models/
```

## Paso 2: Crear Release en GitHub

1. Ve a tu repo: https://github.com/oscarperdomop/grape-disease-classifier
2. Click en **"Releases"** (lado derecho)
3. Click en **"Create a new release"**
4. Llena los datos:

   - **Tag version:** `v1.0.0` (o el número que quieras)
   - **Release title:** `Model Files v1.0.0`
   - **Description:** `ONNX model files for grape disease classification`

5. **Drag & drop** el archivo `models.zip` en la sección "Attach binaries"
6. Click en **"Publish release"**

## Paso 3: Verificar

El script `download_models.py` automáticamente:

1. Busca la última Release
2. Descarga `models.zip`
3. Extrae los modelos a `models/`
4. Se ejecuta durante el build de Docker

## Si algo falla:

- Ve a los **Build Logs** de Render
- Busca líneas con "Model Download Script"
- Verifica que el ZIP tenga la estructura correcta:
  ```
  models.zip
  ├── model_1/
  │   ├── model.onnx
  │   └── config.json
  ├── model_2/
  │   ├── model.onnx
  │   └── config.json
  ...
  ```

## Para actualizar modelos en el futuro:

1. Crea una nueva Release con versión más alta (ej: `v1.0.1`)
2. Sube el ZIP
3. Render automáticamente descargará la última versión en el próximo deploy
