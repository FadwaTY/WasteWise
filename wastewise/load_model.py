# core/model_loader.py
from ultralytics import YOLO

# Chargement du modèle (1 seule fois)
model = YOLO("models/bestmodel.pt")
