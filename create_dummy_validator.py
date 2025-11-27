#!/usr/bin/env python3
"""
Script para crear un modelo validador DUMMY para testing.
Este modelo es muy simple y solo sirve para verificar que el pipeline funciona.
Para producción, necesitas entrenar un modelo real.
"""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto
from pathlib import Path

# Crear directorio
validator_dir = Path("models/validator")
validator_dir.mkdir(parents=True, exist_ok=True)

print("🔨 Creando modelo validador DUMMY...")

# Crear un modelo ONNX simple que:
# - Toma entrada: (1, 3, 224, 224)
# - Retorna salida: (1, 2) - probabilidades binarias

# Input
X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])

# Output
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

# Crear un nodo simple que retorna probabilidades constantes
# Para testing: siempre retorna [0.3, 0.7] (70% confianza en "es hoja de uva")
const_tensor = helper.make_tensor(
    name='const_output',
    data_type=TensorProto.FLOAT,
    dims=[1, 2],
    vals=[0.3, 0.7]  # 30% no es hoja, 70% es hoja
)

identity_node = helper.make_node(
    'Identity',
    inputs=['const_output'],
    outputs=['output']
)

# Crear el grafo
graph_def = helper.make_graph(
    [identity_node],
    'ValidatorModel',
    [X],
    [Y],
    [const_tensor]
)

# Crear el modelo
model_def = helper.make_model(graph_def, producer_name='grape-disease-classifier')
model_def.opset_import[0].version = 12

# Guardar
model_path = validator_dir / "model.onnx"
onnx.save(model_def, str(model_path))

print(f"✅ Modelo guardado en: {model_path}")

# Crear config.json
config = {
    "labels": ["Not Grape Leaf", "Grape Leaf"],
    "classes": ["Not Grape Leaf", "Grape Leaf"]
}

import json
config_path = validator_dir / "config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✅ Config guardado en: {config_path}")

print("\n" + "="*60)
print("⚠️  IMPORTANTE - MODELO DUMMY PARA TESTING")
print("="*60)
print("\nEste modelo DUMMY siempre retorna:")
print("  - 30% confianza en 'No es hoja de uva'")
print("  - 70% confianza en 'Es hoja de uva'")
print("\nComo el threshold es 0.5, TODAS las imágenes serán ACEPTADAS")
print("\nPara testing real, necesitas:")
print("  1. Entrenar un modelo binario real")
print("  2. Usar dataset con hojas de uva vs otras imágenes")
print("  3. Exportar a ONNX")
print("  4. Reemplazar models/validator/model.onnx")
print("\nVer: VALIDATOR_MODEL_GUIDE.md para instrucciones de entrenamiento")
print("="*60)
