# core/inference.py
import numpy as np
import cv2
from wastewise.load_model import model  # on importe le modèle déjà chargé

def run_inference(image_bytes):
    # Convertir bytes → image OpenCV
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Lancer YOLO
    results = model(img)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
            })

    return detections
