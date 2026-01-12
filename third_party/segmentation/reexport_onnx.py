from ultralytics import FastSAM
import torch
import os

model_path = '/home/syl/GICI-IM/third_party/segmentation/FastSAM-s.pt'
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    exit(1)

print(f"Loading model from {model_path}")
model = FastSAM(model_path)

print("Exporting to ONNX (Opset 11, Fixed Size 320x320)...")
# Note: output name is usually derived from input but can be handled
try:
    path = model.export(format='onnx', imgsz=320, opset=10, dynamic=False, simplify=True)
    print(f"Exported to: {path}")
except Exception as e:
    print(f"Export failed: {e}")

