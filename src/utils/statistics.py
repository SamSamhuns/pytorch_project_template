import sys
import logging
from subprocess import call

import torch


def print_cuda_statistics():
    """
    Print statistics of cuda devices
    """
    logger = logging.getLogger("Cuda Statistics")

    logger.info('__Python VERSION: %s', sys.version)
    logger.info('__pyTorch VERSION:  %s', torch.__version__)
    logger.info('__CUDA VERSION')
    try:
        call(["nvcc", "--version"])
    except Exception as excep:
        logger.error("%s: nvcc not found", excep)
    logger.info('__CUDNN VERSION: %s', torch.backends.cudnn.version())
    logger.info('__Number CUDA Devices: %s', torch.cuda.device_count())
    logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('Active CUDA Device: GPU %s', torch.cuda.current_device())
    logger.info('Available devices  %s', torch.cuda.device_count())
    logger.info('Current cuda device  %s', torch.cuda.current_device())


def get_model_params(model: torch.nn.Module):
    """
    Get total number of params in model 
    """
    total_params = sum(torch.numel(p) for p in model.parameters())
    net_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = sum(torch.numel(p) for p in net_parameters)
    return {'Model': model.__class__,
            'Total params': total_params,
            'Trainable params': trainable_params,
            'Non-trainable params': total_params - trainable_params}
