"""Serves a model exported by export.py: POST an image to /predict to get its most likely classes.
Run from the repo root: MODEL_PATH=path/to/model.onnx uvicorn server.app:app --port 8080
"""

import json
import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

# export.py stores the class names and the settings of the eval transform in the ONNX file
session = ort.InferenceSession(os.environ.get("MODEL_PATH", "checkpoints_server/model.onnx"))
metadata = session.get_modelmeta().custom_metadata_map
classes = json.loads(metadata["classes"])
eval_settings = json.loads(metadata["preprocess"])

app = FastAPI(title="Image classifier")


def preprocess(
    image: Image.Image, image_mode: str, resize_size: int, crop_size: int, mean: list, std: list, interpolation: str
) -> np.ndarray:
    """The eval transform of training, in PIL and numpy, as the server has no torchvision: resize the shorter side
    to resize_size, center crop crop_size, then scale to [0, 1] and normalize. Returns a batch of one image
    """
    image = image.convert("RGB").convert(image_mode)  # RGB as inference.py loads images, then L for MNIST's Grayscale
    width, height = image.size
    long_side = int(resize_size * max(width, height) / min(width, height))  # rounded down, as torchvision does
    size = (resize_size, long_side) if width <= height else (long_side, resize_size)
    image = image.resize(size, Image.Resampling[interpolation.upper()])
    left, top = round((image.width - crop_size) / 2), round((image.height - crop_size) / 2)
    image = image.crop((left, top, left + crop_size, top + crop_size))
    pixels = np.asarray(image, dtype=np.float32).reshape(crop_size, crop_size, -1) / 255  # height, width, channels
    pixels = (pixels - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return pixels.transpose(2, 0, 1)[None]


# a def, not an async def: FastAPI runs it in a thread pool, so one request doesn't hold up the others
@app.post("/predict")
def predict(file: UploadFile, top_k: int = 5) -> list[dict]:
    try:
        batch = preprocess(Image.open(file.file), **eval_settings)
    except OSError as excep:  # PIL can't read the file
        raise HTTPException(status_code=400, detail="The file is not an image") from excep
    logits = session.run(None, {session.get_inputs()[0].name: batch})[0][0]
    probs = np.exp(logits - logits.max())  # softmax
    probs /= probs.sum()
    return [{"class": classes[i], "probability": float(probs[i])} for i in probs.argsort()[::-1][:top_k]]
