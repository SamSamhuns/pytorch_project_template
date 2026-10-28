"""Tests for server/app.py"""

import importlib
import io
import json
import sys

import numpy as np
import pytest
import torch
from fastapi import HTTPException, UploadFile
from PIL import Image
from torch import nn

from src.augmentations import imagenet_transforms, mnist_transforms
from src.utils.export_utils import export_onnx

CLASSES = ["ant", "bee", "cat"]


def start_server(settings, tmp_path, monkeypatch):
    """Export a small model with these eval settings and import the server with it. Returns both"""
    channels, size = {"L": 1, "RGB": 3}[settings["image_mode"]], settings["crop_size"]
    # every pixel reaches the output, so a preprocessing that differs from the eval transform changes it
    model = nn.Sequential(nn.Flatten(), nn.Linear(channels * size * size, len(CLASSES)))
    metadata = {"classes": json.dumps(CLASSES), "preprocess": json.dumps(settings)}
    export_onnx(model, torch.randn(2, channels, size, size), str(tmp_path / "model.onnx"), metadata)

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.onnx"))
    monkeypatch.delitem(sys.modules, "server.app", raising=False)
    return model, importlib.import_module("server.app")  # loads the model on import


def upload(data: bytes) -> UploadFile:
    return UploadFile(io.BytesIO(data))


@pytest.mark.parametrize("transforms_fn", [mnist_transforms, imagenet_transforms])
def test_server_predicts_like_pytorch(transforms_fn, tmp_path, monkeypatch):
    """The server repeats the eval transform without torchvision, so its probabilities must equal PyTorch's"""
    _, eval_transform, eval_settings = transforms_fn()
    model, server = start_server(eval_settings, tmp_path, monkeypatch)
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (300, 400, 3), dtype=np.uint8))
    png = io.BytesIO()
    image.save(png, format="PNG")

    predictions = server.predict(upload(png.getvalue()), top_k=len(CLASSES))

    with torch.inference_mode():
        expected = model(eval_transform(image).unsqueeze(0)).softmax(dim=1)[0].numpy()
    probabilities = {prediction["class"]: prediction["probability"] for prediction in predictions}
    np.testing.assert_allclose([probabilities[name] for name in CLASSES], expected, atol=1e-5)
    assert [prediction["class"] for prediction in predictions] == [CLASSES[i] for i in expected.argsort()[::-1]]


def test_server_rejects_files_that_are_not_images(tmp_path, monkeypatch):
    _, _, eval_settings = mnist_transforms()
    _, server = start_server(eval_settings, tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as excinfo:
        server.predict(upload(b"not an image"))
    assert excinfo.value.status_code == 400
