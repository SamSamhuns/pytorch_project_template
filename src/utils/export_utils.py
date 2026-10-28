"""PyTorch export utils."""
from abc import ABC, abstractmethod
from logging import Logger
from timeit import Timer

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn.modules import Module


class ExportStrategy(ABC):
    """Base class for export strategy.

    logger: Optional logger object to log instead of printing
    """

    def __init__(self, logger: Logger = None) -> None:
        super().__init__()
        self.logger = logger

    @abstractmethod
    def export(self, model: Module, path: str, sample_in: torch.Tensor):
        """Exports pytorch model to implemented export mode."""

    @abstractmethod
    def test_inference(self, model: Module, path: str, sample_in: torch.Tensor):
        """Tests exported model output vs the original torch model output for similarity."""


def torch_inference(model, sample_in):
    """Pytorch based inference."""
    with torch.no_grad():
        return model(sample_in).cpu().numpy()


def onnx_inference(ort_sess, sample_in):
    """ONNX based inference"""
    ort_inputs = {ort_sess.get_inputs()[0].name: sample_in.cpu().numpy()}
    ort_outs = ort_sess.run(None, ort_inputs)
    return ort_outs


def onnx_inference_check(
        model: Module, onnx_model_path: str, sample_in: torch.Tensor, logger: Logger = None) -> None:
    """Test exported ONNX model and inference.
    
    Currently only supports CPU based export check
    Returns True if the ONNX model's output matches the PyTorch model's output, False otherwise.
    """
    # only cpu mode supported for now
    model = model.to(torch.device("cpu"))
    sample_in = sample_in.cpu()
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    ort_sess = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

    ort_out = onnx_inference(ort_sess, sample_in)[0]
    torch_out = torch_inference(model, sample_in)

    # Compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-03, atol=1e-05)

    # time torch and onnx inf runs
    runs = 32
    torch_tm = Timer(lambda: torch_inference(model, sample_in)).timeit(number=runs)
    ort_tm = Timer(lambda: onnx_inference(ort_sess, sample_in)).timeit(number=runs)

    pt_tm_msg = f"Pytorch average inf time for tensor {sample_in.shape} = {torch_tm/runs:.3f}s"
    onnx_tm_msg = f"ONNX average inf time for tensor {sample_in.shape} = {ort_tm/runs:.3f}s"
    exp_msg = "\u2705 Exported ONNX model results match that of PyTorch model!"
    if logger:
        logger.info(pt_tm_msg)
        logger.info(onnx_tm_msg)
        logger.info(exp_msg)
    else:
        print(pt_tm_msg, onnx_tm_msg)
        print(exp_msg)


class ONNXTSExportStrategy(ExportStrategy):
    """ONNX torchscript export logic."""

    def export(self, model: Module, path: str, sample_in: torch.Tensor):
        # only cpu export mode supported as of now
        model = model.to(torch.device("cpu"))
        sample_in = sample_in.cpu()
        torch.onnx.export(
            model, sample_in, path, export_params=True,
            opset_version=17, do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

    def test_inference(self, model: Module, path: str, sample_in: torch.Tensor):
        return onnx_inference_check(model, path, sample_in, self.logger)


class ONNXDynamoExportStrategy(ExportStrategy):
    """ONNX dynamo export logic"""

    def export(self, model: Module, path: str, sample_in: torch.Tensor):
        # Only CPU export mode supported as of now
        model = model.to(torch.device("cpu"))
        sample_in = sample_in.cpu()

        # Use dynamo_export, not export with dynamo=True
        export_output = torch.onnx.export(
            model,
            sample_in,
            dynamo=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        # Save the exported model
        export_output.save(path)

    def test_inference(self, model: Module, path: str, sample_in: torch.Tensor):
        return onnx_inference_check(model, path, sample_in, self.logger)
