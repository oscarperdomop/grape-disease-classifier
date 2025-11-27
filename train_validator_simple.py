#!/usr/bin/env python3
"""
Script simple para entrenar un modelo validador binario.
Usa transfer learning con ResNet18 pre-entrenado.

Requisitos:
  pip install torch torchvision onnx onnxruntime pillow numpy

Uso:
  1. Prepara dataset en: dataset/grape_leaf/ y dataset/not_grape_leaf/
  2. Ejecuta: python train_validator_simple.py
  3. Modelo se guarda en: models/validator/model.onnx
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import onnx
import numpy as np

# Configuración
DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("models/validator")
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Usando device: {DEVICE}")

# Verificar dataset
if not DATASET_DIR.exists():
    print(f"❌ Dataset no encontrado en {DATASET_DIR}")
    print("\nCrea la estructura:")
    print("  dataset/")
    print("  ├── grape_leaf/")
    print("  │   ├── image1.jpg")
    print("  │   ├── image2.jpg")
    print("  │   └── ...")
    print("  └── not_grape_leaf/")
    print("      ├── image1.jpg")
    print("      ├── image2.jpg")
    print("      └── ...")
    sys.exit(1)

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset
print(f"📂 Cargando dataset desde {DATASET_DIR}...")
train_dataset = datasets.ImageFolder(str(DATASET_DIR), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"✅ Dataset cargado: {len(train_dataset)} imágenes")
print(f"   Clases: {train_dataset.classes}")

# Modelo
print("\n🧠 Creando modelo...")
model = models.resnet18(pretrained=True)

# Congelar capas pre-entrenadas
for param in model.parameters():
    param.requires_grad = False

# Reemplazar última capa para clasificación binaria
model.fc = nn.Linear(512, 2)

# Descongelar última capa
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)
print(f"✅ Modelo creado en {DEVICE}")

# Criterio y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# Entrenamiento
print("\n🎓 Entrenando...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Estadísticas
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

print("✅ Entrenamiento completado")

# Exportar a ONNX
print("\n📦 Exportando a ONNX...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model.eval()
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

onnx_path = OUTPUT_DIR / "model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    input_names=['input'],
    output_names=['output'],
    opset_version=12,
    do_constant_folding=True,
    verbose=False
)

print(f"✅ Modelo exportado a {onnx_path}")

# Crear config.json
import json
config = {
    "labels": ["Not Grape Leaf", "Grape Leaf"],
    "classes": ["Not Grape Leaf", "Grape Leaf"],
    "description": "Binary classifier for grape leaf validation"
}

config_path = OUTPUT_DIR / "config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✅ Config guardado en {config_path}")

# Validar modelo ONNX
print("\n✔️ Validando modelo ONNX...")
try:
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✅ Modelo ONNX válido")
except Exception as e:
    print(f"❌ Error validando modelo: {e}")
    sys.exit(1)

# Test rápido
print("\n🧪 Test rápido...")
import onnxruntime as ort
from PIL import Image

sess = ort.InferenceSession(str(onnx_path))
input_name = sess.get_inputs()[0].name

# Crear imagen dummy
test_img = Image.new('RGB', (224, 224), color='red')
test_arr = np.array(test_img).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
test_arr = (test_arr - mean) / std
test_arr = np.transpose(test_arr, (2, 0, 1))
test_arr = np.expand_dims(test_arr, axis=0)

output = sess.run(None, {input_name: test_arr})[0]
probs = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)

print(f"  No es hoja: {probs[0][0]:.2%}")
print(f"  Es hoja: {probs[0][1]:.2%}")

print("\n" + "="*60)
print("✅ MODELO VALIDADOR ENTRENADO Y LISTO")
print("="*60)
print(f"\nModelo guardado en: {onnx_path}")
print(f"Config guardado en: {config_path}")
print("\nPróximos pasos:")
print("  1. Reinicia el backend")
print("  2. Prueba con: curl http://localhost:8000/validator")
print("  3. Sube una imagen: curl -X POST -F 'file=@image.jpg' http://localhost:8000/predict")
print("="*60)
