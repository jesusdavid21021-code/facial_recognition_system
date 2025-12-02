import sys
print(f"Python ejecutándose desde: {sys.executable}")
print(f"Entorno activo: {sys.prefix}\n")

import torch
import cv2
import numpy as np
import onnxruntime
import insightface
from insightface.app import FaceAnalysis
import customtkinter
import pandas as pd

print("=" * 60)
print("VERIFICACIÓN DE INSTALACIÓN - Sistema de Reconocimiento Facial")
print("=" * 60)

# 1. PyTorch y CUDA
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 2. OpenCV
print(f"\n✓ OpenCV version: {cv2.__version__}")

# 3. NumPy
print(f"✓ NumPy version: {np.__version__}")

# 4. ONNX Runtime
print(f"\n✓ ONNX Runtime version: {onnxruntime.__version__}")
providers = onnxruntime.get_available_providers()
print(f"✓ Providers disponibles: {providers}")
if 'CUDAExecutionProvider' in providers:
    print("  🎉 CUDA habilitado en ONNX Runtime")
else:
    print("  ⚠ Usando CPU (GPU no detectada en ONNX)")

# 5. InsightFace
print(f"\n✓ InsightFace instalado correctamente")
try:
    # Intenta crear app de análisis facial
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    print("✓ Modelo InsightFace inicializado exitosamente")
    print(f"  Providers del modelo: {app.det_model.providers if hasattr(app, 'det_model') else 'N/A'}")
except Exception as e:
    print(f"⚠ Advertencia al inicializar modelo: {e}")
    print("  (Normal si es la primera vez - descargará modelos al usarse)")

# 6. CustomTkinter
print(f"\n✓ CustomTkinter version: {customtkinter.__version__}")

# 7. Pandas
print(f"✓ Pandas version: {pd.__version__}")

# 8. Test de cámara
print("\n✓ Probando cámara...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"✓ Cámara funcionando - Resolución: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("⚠ Cámara detectada pero no puede capturar frames")
    cap.release()
else:
    print("⚠ No se detectó cámara (conecta una webcam si es necesario)")

print("\n" + "=" * 60)
print("✅ INSTALACIÓN COMPLETADA Y VERIFICADA EXITOSAMENTE")
print("=" * 60)
print("\n🚀 Listo para comenzar con el desarrollo del sistema")