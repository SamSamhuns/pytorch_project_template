"""
Tests for src.utils.export_utils
"""
import os
import numpy as np

import pytest
from tests.conftest import PYTEST_TEMP_ROOT
from src.utils.export_utils import (
    torch_inference, onnx_inference,
    ONNXTSExportStrategy, ONNXDynamoExportStrategy, TSTraceExportStrategy, TSScriptExportStrategy)


def test_torch_inference(simple_1d_conv_model, sample_tensor):
    """Perform inference"""
    output = torch_inference(simple_1d_conv_model, sample_tensor)
    # Check output type and shape
    assert isinstance(output, np.ndarray), "Output should be a numpy array"
    assert output.shape == (32, 10), "Output shape is incorrect"


def test_onnx_inference(onnx_session, sample_tensor):
    """Perform inference"""
    output = onnx_inference(onnx_session, sample_tensor)
    # Since onnx_inference returns a list of outputs, we check the first item
    assert isinstance(output, list), "Output should be a list"
    assert len(output) == 1, "Output list should contain one item"
    assert output[0].shape == (32, 10), "Output shape is incorrect"


def test_inference_compatibility(simple_1d_conv_model, onnx_session, sample_tensor):
    """Test pytorch and onnx inference compatibility"""
    # Get outputs from both PyTorch and ONNX models
    torch_output = torch_inference(simple_1d_conv_model, sample_tensor)
    onnx_output = onnx_inference(onnx_session, sample_tensor)[0]
    # Compare outputs
    np.testing.assert_allclose(
        torch_output, onnx_output, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize("strategy_class, export_file_extn", [
    (ONNXTSExportStrategy, ".onnx"),
    (ONNXDynamoExportStrategy, ".onnx"),
    (TSTraceExportStrategy, ".pt"),
    (TSScriptExportStrategy, ".pt")
])
def test_export(setup_model_and_logger, sample_tensor, strategy_class, export_file_extn):
    model, logger = setup_model_and_logger
    export_path = PYTEST_TEMP_ROOT + f"/temp_model{export_file_extn}"
    strategy = strategy_class(logger)
    strategy.export(model, export_path, sample_tensor)
    assert os.path.exists(export_path), "Export file was not created"


@pytest.mark.parametrize("strategy_class, export_file_extn", [
    (ONNXTSExportStrategy, ".onnx"),
    (ONNXDynamoExportStrategy, ".onnx"),
    (TSTraceExportStrategy, ".pt"),
    (TSScriptExportStrategy, ".pt")
])
@pytest.mark.order(after=["test_export"])
def test_inference(setup_model_and_logger, sample_tensor, strategy_class, export_file_extn):
    """Test exported models inference"""
    model, logger = setup_model_and_logger
    file_path = PYTEST_TEMP_ROOT + f"/temp_model{export_file_extn}"
    strategy = strategy_class(logger)
    strategy.export(model, file_path, sample_tensor)
    assert strategy.test_inference(
        model, file_path, sample_tensor) is None, "Inference test failed"
