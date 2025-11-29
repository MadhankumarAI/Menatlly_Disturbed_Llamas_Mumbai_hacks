# server/serve_fastapi.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
from infer import Recognizer

app = FastAPI()

# load model once at startup
MODEL_CKPT = "./models/recognition.pth"
recognizer = Recognizer(MODEL_CKPT)

class PredictRequest(BaseModel):
    keypoints: list  # list of frames; each frame is flat list of floats length input_dim

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        arr = np.array(req.keypoints, dtype=np.float32)
        label, conf, probs = recognizer.predict(arr)
        return {"label": label, "confidence": conf}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("serve_fastapi:app", host="0.0.0.0", port=8000, workers=1)
