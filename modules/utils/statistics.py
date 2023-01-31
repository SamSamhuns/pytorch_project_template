import sys
import logging
from subprocess import call

import torch


def print_cuda_statistics():
    logger = logging.getLogger("Cuda Statistics")


    logger.info('__Python VERSION: %s', sys.version)
    logger.info('__pyTorch VERSION:  %s', torch.__version__)
    logger.info('__CUDA VERSION')
    try:
        call(["nvcc", "--version"])
    except Exception as e:
        logger.error(f"{e}: nvcc not found")
    logger.info('__CUDNN VERSION: %s', torch.backends.cudnn.version())
    logger.info('__Number CUDA Devices: %s', torch.cuda.device_count())
    logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('Active CUDA Device: GPU %s', torch.cuda.current_device())
    logger.info('Available devices  %s', torch.cuda.device_count())
    logger.info('Current cuda device  %s', torch.cuda.current_device())
