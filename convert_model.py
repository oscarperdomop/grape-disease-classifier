#!/usr/bin/env python3
"""
Convertir modelo PyTorch a ONNX para el validador
"""
import torch
import torch.nn as nn
from torchvision import models
import onnx
from pathlib import Path
import json

# Configuración
PYTORCH_MODEL = "modelo_resnet50v2_entrenado.pth"
OUTPUT_DIR = Path("models/validator")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("🔄 Convirtiendo modelo PyTorch a ONNX...")

# Crear modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, 2)

# Cargar pesos
print(f"📂 Cargando: {PYTORCH_MODEL}")
checkpoint = torch.load(PYTORCH_MODEL, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

# Convertir a ONNX
dummy_input = torch.randn(1, 3, 224, 224).to(device)
onnx_path = OUTPUT_DIR / "model.onnx"

print(f"⚙️  Exportando a: {onnx_path}")
torch.onnx.export(
    model, dummy_input, str(onnx_path),
    input_names=['input'],
    output_names=['output'],
    opset_version=12,
    do_constant_folding=True
)

# Validar
onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)
print("✅ Modelo ONNX válido")

# Crear config
config = {
    "labels": ["Not Grape Leaf", "Grape Leaf"],
    "classes": ["Not Grape Leaf", "Grape Leaf"]
}
config_path = OUTPUT_DIR / "config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✅ Guardado en: {OUTPUT_DIR}")
print(f"   - model.onnx")
print(f"   - config.json")
print("\n✅ Ahora reinicia el backend y prueba")
