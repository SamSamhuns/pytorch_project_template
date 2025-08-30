import logging
import sys
from subprocess import call

import torch
from torch.utils import data


def print_cuda_statistics():
    """Print statistics of cuda devices."""
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
    """Get total number of params in model."""
    total_params = sum(torch.numel(p) for p in model.parameters())
    net_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = sum(torch.numel(p) for p in net_parameters)
    return {'Model': model.__class__,
            'Total params': total_params,
            'Trainable params': trainable_params,
            'Non-trainable params': total_params - trainable_params}


def get_img_dset_mean_std(dataset: data.Dataset, method: str = "online") -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the channel-wise mean and std dev of the train data using an online two-pass method.

    or an offline one-pass method which the entire data is loaded in memory.
    Results match for both methods within a difference of 1e-4

    Args:
        dataset: Dataset which should return data,_ when iterated over.
                 The data should be an image tensor of shape [3, H, W].
        method: offline or online method

    """
    assert method in {"online", "offline"}, "method can only be set to `online` or `offline`"
    if method == "online":
        # Initialize accumulators
        mean = torch.zeros(3)  # Assuming 3 channels (RGB)
        var = torch.zeros(3)
        n_pixels = 0

        # First pass: Calculate the mean
        for sample, _ in dataset:
            sample_flat = sample.view(3, -1)  # Flatten to [3, H*W]
            n_pixels += sample_flat.size(1)  # Total pixels per channel
            # Sum of all pixel values for each channel
            mean += sample_flat.sum(dim=1)
        mean /= n_pixels  # Normalize to get mean

        # Second pass: Calculate variance
        for sample, _ in dataset:
            sample_flat = sample.view(3, -1)  # Flatten to [3, H*W]
            # Sum squared differences for each channel
            var += ((sample_flat - mean.view(3, 1)) ** 2).sum(dim=1)
        var /= n_pixels  # Normalize variance

        std = torch.sqrt(var)
    else:  # offline method
        print("WARNING: Offline method is memory intensive and loads all data in memory")
        all_data = torch.cat([sample.view(3, -1)
                             for sample, _ in dataset], dim=1)
        mean = all_data.mean(dim=1)
        std = all_data.std(dim=1)
    return mean, std
