# TODO: Import your package, replace this by explicit imports of what you need
from wastewise.new import run_inference
from wastewise.new import test_dataset
from wastewise.llm_reco import get_recycling_advice


from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.get("/predict")
def get_predict(input_one: float,
            input_two: float):
    # TODO: Do something with your input
    # i.e. feed it to your model.predict, and return the output
    # For a dummy version, just return the sum of the two inputs and the original inputs
    prediction = float(input_one) + float(input_two)
    return {
        'prediction': prediction,
        'inputs': {
            'input_one': input_one,
            'input_two': input_two
        }
    }

# @app.post("/detect")
# async def detect(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     result = run_inference(image_bytes)
#     return {"detections": result}

# @app.post("/detect")
# async def detect(file: UploadFile = File(...)):
#     image_bytes = await file.read()

#     # 1. YOLO inference
#     detections = run_inference(image_bytes)

#     # 2. Ajouter les recommandations LLM
#     for d in detections:
#         class_name = d["class_name"]
#         d["recycling_advice"] = get_recycling_advice(class_name)

#     return {"detections": detections}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Gérer la lecture du fichier
    try:
        image_bytes = await file.read()
    except Exception as e:
        # Gérer l'erreur de lecture du fichier
        raise HTTPException(status_code=400, detail="Impossible de lire le fichier téléchargé.")

    # 1. YOLO inference (Assurez-vous que cette fonction est stable)
    detections = run_inference(image_bytes)

    # 2. Ajouter les recommandations LLM
    for d in detections:
        class_name = d.get("class_name")
        if class_name:
            # Appel à la fonction corrigée
            d["recycling_advice"] = get_recycling_advice(class_name)
        else:
            d["recycling_advice"] = "Classification incomplète."

    return {"detections": detections}

@app.get("/test_all")
def test_all():
    return test_dataset()

# @app.get("/test_one")
# def test_one():
#     test_folder = "/test"
#     files = [f for f in os.listdir(test_folder)
#              if f.lower().endswith((".jpg", ".png", ".jpeg"))]

#     filename = files[0]
#     print("Testing:", filename)

#     with open(os.path.join(test_folder, filename), "rb") as f:
#         image_bytes = f.read()

#     return run_inference(image_bytes)
