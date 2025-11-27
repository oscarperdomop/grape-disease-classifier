# ✅ Activar Validador - Pasos Rápidos

Tu modelo validador está entrenado pero NO está activo. Aquí cómo activarlo:

## 🚀 Pasos

### 1. Convertir a ONNX (1 minuto)

```bash
python convert_model.py
```

**Qué hace:**
- Lee: `modelo_resnet50v2_entrenado.pth`
- Convierte a: `models/validator/model.onnx`
- Crea: `models/validator/config.json`

**Resultado:**
```
✅ Guardado en: models/validator
   - model.onnx
   - config.json
```

### 2. Reiniciar Backend

```bash
# Si estaba corriendo, presiona Ctrl+C
# Luego:
uvicorn backend.app:app --reload
```

Deberías ver en los logs:
```
🔍 Loading validator model...
✅ Validator model loaded successfully
```

### 3. Probar

**Con imagen válida (hoja de uva):**
```bash
curl -X POST -F "file=@hoja.jpg" http://localhost:8000/predict?model_id=model_1
```

Resultado: Predicción normal (Healthy, ESCA, etc.)

**Con imagen inválida (carro, persona, etc.):**
```bash
curl -X POST -F "file=@carro.jpg" http://localhost:8000/predict?model_id=model_1
```

Resultado:
```json
{
  "error": "Image is not in scope",
  "message": "The image does not appear to be a grape leaf",
  "validation_confidence": 0.12,
  "note": "Please provide an image of a grape leaf for analysis"
}
```

## ✅ Verificación Rápida

```bash
# Ver si validador está cargado
curl http://localhost:8000/validator
```

Debería mostrar:
```json
{
  "validator_enabled": true,
  "validator_model_id": "validator",
  "validator_threshold": 0.5,
  "description": "Binary classifier that validates if image is a grape leaf"
}
```

## 🎯 Resumen

| Paso | Comando | Tiempo |
|------|---------|--------|
| 1. Convertir | `python convert_model.py` | 1 min |
| 2. Reiniciar | `Ctrl+C` + `uvicorn...` | 10 seg |
| 3. Probar | `curl http://localhost:8000/validator` | 5 seg |

**Total: ~2 minutos**

## ❓ ¿Qué pasaba antes?

❌ **Antes**: Imagen X → Clasificada como ESCA (sin validar)
✅ **Ahora**: Imagen X → Rechazada como "No es hoja de uva"

## 🚀 Desplegar a Producción

Una vez que funcione localmente:

```bash
git add models/validator/
git commit -m "feat: Add validator model (ONNX)"
git push origin main
```

Render desplegará automáticamente en 2-3 minutos.

---

¿Necesitas ayuda con algún paso?
