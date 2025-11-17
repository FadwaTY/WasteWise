import base64
import io
import os
from typing import Dict, List, Optional
from .reco import recommend

from PIL import Image, ImageDraw
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# ---- per-class thresholds (optional) ----
DEFAULT_CLASS_CONF = {
    "BIODEGRADABLE": 0.22,
    "CARDBOARD":     0.30,
    "GLASS":         0.30,
    "METAL":         0.28,
    "PLASTIC":       0.35,
    "PAPER":         0.70,
}

# ---- device / app setup ----
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

APP_NAME = "WasteWise API"
MODEL_PATH = os.getenv("WW_MODEL_PATH", "models/best.pt")

app = FastAPI(title=APP_NAME)

# CORS for local Streamlit and future UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Load model once at startup
model = YOLO(MODEL_PATH)
# ultralytics handles device inside predict, but we keep a flag for /healthz
_ = model  # just to be explicit


# ---- MODELS for responses ----
class Detection(BaseModel):
    class_name: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]

class PredictResponse(BaseModel):
    counts: Dict[str, int]
    summary: str
    annotated_image_b64: str
    detections: List[Detection]
    recommendation: Optional[str] = None

class RecommendIn(BaseModel):
    counts: Dict[str, int]
    detections: List[Detection] = []

@app.post("/recommend")
def recommend_endpoint(inp: RecommendIn):
    from .reco import recommend
    text = recommend(inp.counts, [d.model_dump() for d in inp.detections])
    return {"recommendation": text}


@app.get("/healthz")
def healthz():
    names = getattr(model, "names", None)
    if names is None:
        # some versions expose on results, but usually model.names exists
        names = {}
    health_dic = {
        "device": DEVICE,
        "num_classes": len(names),
        "classes": names,
        "llm_provider": os.getenv("WW_LLM_PROVIDER"),
        "llm_model": os.getenv("WW_LLM_MODEL"),
        "ollama_url": os.getenv("WW_OLLAMA_URL"),
        }
    status = ""
    if health_dic.get("num_classes")!=6 or not bool(health_dic.get("classes")):
        status = "bad"
    elif len(health_dic)==6:
        status = "ok"
    else:
        status = "partial"
    health_dic["status"]=status
    return health_dic

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.25),      # use Form so we can send alongside file
    iou: float = Form(0.45),
    llm: bool = Form(True),        # NEW
):
    # Read and convert to PIL
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")

    # Run inference
    results = model.predict(
        image,
        conf=conf,
        iou=iou,
        verbose=False,
        device=0 if DEVICE == "cuda" else "cpu",
    )
    r = results[0]

    # Names mapping (index -> class)
    names = r.names if hasattr(r, "names") else getattr(model, "names", {})

    xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    cls  = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.array([], dtype=int)
    confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.array([], dtype=float)

    # Keep only boxes that pass (per-class) threshold
    kept = []
    for box, c, s in zip(xyxy, cls, confs):
        cname = names.get(int(c), str(int(c))) if isinstance(names, dict) else str(int(c))
        thr = DEFAULT_CLASS_CONF.get(str(cname).upper(), conf)  # fallback to global slider
        if float(s) >= float(thr):
            kept.append((box, str(cname), float(s)))

    # Detections list (from kept only)
    detections: List[Detection] = [
        Detection(class_name=cname, confidence=score, box=[float(x) for x in box.tolist()])
        for (box, cname, score) in kept
    ]

    # Counts & summary (from kept only)
    counts: Dict[str, int] = {}
    for _, cname, _ in kept:
        counts[cname] = counts.get(cname, 0) + 1

    summary = ", ".join([f"{v} {k.upper()}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]) \
            if counts else "No waste detected"

    # Annotate image using ONLY kept boxes
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # deterministic per-class colors (simple hash)
    def _color_for(name: str):
        h = abs(hash(name)) % 255
        return (50 + (h * 3) % 205, 50 + (h * 5) % 205, 50 + (h * 7) % 205)

    for box, cname, score in kept:
        x1, y1, x2, y2 = map(int, box.tolist())
        color = _color_for(cname)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        label = f"{cname} {score:.2f}"
        # text box
        tw, th = draw.textlength(label), 14  # PIL default font approx height
        bg_w = int(tw) + 6
        bg_h = th + 6
        y_top = max(0, y1 - bg_h)
        draw.rectangle([(x1, y_top), (x1 + bg_w, y_top + bg_h)], fill=color)
        draw.text((x1 + 3, y_top + 3), label, fill=(0, 0, 0))

    # Encode annotated image
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    rec_text = None
    if llm:
        from .reco import recommend
        rec_text = recommend(
            counts=counts,
            detections=[d.model_dump() for d in detections]
        )

    return PredictResponse(
        counts=counts,
        summary=summary,
        annotated_image_b64=data_url,
        detections=detections,
        recommendation=rec_text,
    )
