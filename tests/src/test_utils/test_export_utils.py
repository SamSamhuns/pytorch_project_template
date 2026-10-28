"""Tests for src.utils.export_utils."""

import onnxruntime as ort
import torch

from src.utils.export_utils import export_onnx


def test_export_onnx(simple_2d_conv_model, tmp_path):
    """The ONNX model takes any batch size and holds the metadata; export_onnx checks its outputs itself"""
    path = str(tmp_path / "model.onnx")
    export_onnx(simple_2d_conv_model, torch.randn(2, 3, 100, 100), path, metadata={"classes": '["ant", "bee"]'})

    session = ort.InferenceSession(path)
    assert session.get_modelmeta().custom_metadata_map["classes"] == '["ant", "bee"]'
    (output,) = session.run(None, {session.get_inputs()[0].name: torch.randn(5, 3, 100, 100).numpy()})
    assert output.shape == (5, 10)
