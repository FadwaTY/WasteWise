# core/inference.py
import numpy as np
import cv2
import os
from wastewise.load_model import model  # modèle YOLO chargé

# Liste des classes dans l'ordre EXACT du modèle
CLASS_NAMES = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']

def run_inference(image_bytes):
    # Convertir bytes → image OpenCV
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Lancer YOLO
    results = model(img)

    detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            detections.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
            })

    return detections


def test_dataset(test_folder="/home/assia/code/FadwaTY/WasteWise/wastewise/test/images"):
    results_all = []

    for filename in os.listdir(test_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):

            # Lire l’image comme bytes
            image_path = os.path.join(test_folder, filename)
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            # Exécuter la détection
            detections = run_inference(image_bytes)

            results_all.append({
                "filename": filename,
                "detections": detections
            })

    return results_all
