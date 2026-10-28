"""ONNX export"""

import numpy as np
import onnxruntime as ort
import torch
from torch import nn


def export_onnx(model: nn.Module, sample: torch.Tensor, path: str, metadata: dict[str, str]) -> None:
    """Export a CPU model to ONNX at path, for any batch size, with metadata such as the class names.
    Raises an AssertionError if ONNX Runtime's outputs for sample differ from the model's
    """
    model.eval()
    # the dynamo exporter, which torch.onnx.export defaults to since torch 2.9; TorchScript export is deprecated
    program = torch.onnx.export(model, (sample,), dynamo=True, dynamic_shapes=({0: "batch"},), verbose=False)
    program.model.metadata_props.update(metadata)
    program.save(path)

    session = ort.InferenceSession(path)
    with torch.inference_mode():
        expected = model(sample).numpy()
    actual = session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)
