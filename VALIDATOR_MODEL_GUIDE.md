# 🔍 Validator Model Guide

## ¿Qué es el Validator?

Un **modelo binario** que valida si una imagen es una **hoja de uva** antes de procesarla con los clasificadores de enfermedad.

### Flujo de Predicción

```
1. Usuario sube imagen
   ↓
2. VALIDATOR MODEL → ¿Es hoja de uva?
   ├─ SÍ → Continuar a clasificadores de enfermedad
   └─ NO → Retornar error "Image is not in scope"
   ↓
3. Cargar modelo de enfermedad (model_1, model_2, model_3, model_4)
   ↓
4. Clasificar enfermedad
   ↓
5. Retornar resultados
```

---

## 🎯 Objetivo del Validator

Rechazar imágenes que **NO sean hojas de uva**:

- ❌ Fotos de carros
- ❌ Fotos de personas
- ❌ Fotos de edificios
- ❌ Fotos de otras plantas
- ❌ Imágenes genéricas

Aceptar imágenes que **SÍ sean hojas de uva**:

- ✅ Hojas sanas
- ✅ Hojas con enfermedades
- ✅ Hojas parciales
- ✅ Hojas en diferentes ángulos

---

## 📊 Estructura del Modelo

### Entrada
- **Imagen**: 224x224 píxeles (RGB)
- **Normalización**: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Salida
- **Clase 0**: "No es hoja de uva" (probabilidad)
- **Clase 1**: "Es hoja de uva" (probabilidad)

### Threshold
- **Default**: 0.5 (50% confianza)
- **Configurable**: `VALIDATOR_MODEL_THRESHOLD` en `app.py`

---

## 🏗️ Cómo Entrenar el Validator

### Opción 1: Usar Modelo Pre-entrenado (Recomendado)

Si ya tienes un modelo binario ONNX:

1. Crea carpeta: `models/validator/`
2. Coloca el modelo: `models/validator/model.onnx`
3. (Opcional) Agrega config: `models/validator/config.json`

```json
{
  "labels": ["Not Grape Leaf", "Grape Leaf"],
  "classes": ["Not Grape Leaf", "Grape Leaf"]
}
```

### Opción 2: Entrenar desde Cero

#### Paso 1: Preparar Dataset

```
dataset/
├── grape_leaf/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (100-500 imágenes de hojas de uva)
└── not_grape_leaf/
    ├── car1.jpg
    ├── person1.jpg
    ├── building1.jpg
    └── ... (100-500 imágenes de otras cosas)
```

#### Paso 2: Script de Entrenamiento

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, ImageFolder
import onnx
import onnxruntime

# Configuración
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = ImageFolder('dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modelo
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)  # Binary classification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Exportar a ONNX
dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(
    model,
    dummy_input,
    'models/validator/model.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=12
)

print("✅ Modelo exportado a models/validator/model.onnx")
```

#### Paso 3: Validar Modelo ONNX

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Cargar modelo
sess = ort.InferenceSession('models/validator/model.onnx')

# Cargar imagen de prueba
img = Image.open('test_image.jpg').resize((224, 224))
arr = np.array(img).astype(np.float32) / 255.0

# Normalizar
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
arr = (arr - mean) / std
arr = np.transpose(arr, (2, 0, 1))
arr = np.expand_dims(arr, axis=0)

# Predicción
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: arr})[0]
probs = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)

print(f"No es hoja de uva: {probs[0][0]:.2%}")
print(f"Es hoja de uva: {probs[0][1]:.2%}")
```

---

## 🔧 Configuración en app.py

### Ajustar Threshold

```python
# En app.py, línea ~56
VALIDATOR_MODEL_THRESHOLD = 0.5  # Cambiar según necesidad

# Valores recomendados:
# 0.3 - Más permisivo (acepta más imágenes)
# 0.5 - Balanceado (default)
# 0.7 - Más estricto (rechaza más imágenes)
```

### Deshabilitar Validator (Opcional)

Si no tienes modelo validador, el sistema funciona sin él:

```python
# El validator simplemente no se carga
# y todas las imágenes se aceptan
```

---

## 📝 Respuestas del API

### Imagen Válida (Hoja de Uva)

```json
{
  "predictions": [
    {
      "label": "Healthy",
      "index": 0,
      "score": 0.95
    }
  ]
}
```

### Imagen Inválida (No es Hoja de Uva)

```json
{
  "error": "Image is not in scope",
  "message": "The image does not appear to be a grape leaf",
  "validation_confidence": 0.23,
  "note": "Please provide an image of a grape leaf for analysis"
}
```

---

## 📊 Métricas Recomendadas

Para evaluar el validator:

```python
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Después de predicciones en test set
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Métricas
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.2%}")

# Objetivo: Accuracy > 95%, Recall > 90%
```

---

## 🚀 Deployment

### 1. Entrenar Modelo Localmente

```bash
python train_validator.py
```

### 2. Crear Release en GitHub

```bash
# Crear carpeta con modelo
mkdir -p models/validator
cp model.onnx models/validator/
cp config.json models/validator/

# Commit
git add models/validator/
git commit -m "Add validator model"
git push origin main

# Crear release con archivo ZIP
# En GitHub: Releases → Create Release → Upload models.zip
```

### 3. Render Descargará Automáticamente

El script `download_models.py` descargará el validator junto con otros modelos.

---

## ✅ Testing

### Endpoint de Info

```bash
curl https://your-api.com/validator
```

Respuesta:
```json
{
  "validator_enabled": true,
  "validator_model_id": "validator",
  "validator_threshold": 0.5,
  "description": "Binary classifier that validates if image is a grape leaf",
  "usage": "Automatically runs before disease classification"
}
```

### Probar con Imagen Válida

```bash
curl -X POST -F "file=@grape_leaf.jpg" \
  https://your-api.com/predict?model_id=model_1
```

### Probar con Imagen Inválida

```bash
curl -X POST -F "file=@car.jpg" \
  https://your-api.com/predict?model_id=model_1
```

Debería retornar:
```json
{
  "error": "Image is not in scope",
  "message": "The image does not appear to be a grape leaf",
  "validation_confidence": 0.15
}
```

---

## 📚 Referencias

- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [ONNX Runtime Python](https://onnxruntime.ai/docs/get-started/with-python.html)
- [ImageNet Normalization](https://pytorch.org/vision/stable/models.html)
