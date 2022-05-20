# Required Keys for Configuration JSON file

An example Configuration file is provided at `classifier_cpu_config.json`

## Required top level Keys

    'name'
    "seed"
    'use_cuda'            # set to true for gpu training
    'cudnn_deterministic' # for repeating results together with seed
    'cudnn_benchmark'     # set to true for faster training with gpu
    'gpu_device'          # cuda device list for single/multi gpu training
    'use_amp'             # automatic mixed precision training for faster train
    'arch'
    'dataset'
    'dataloader'
    'optimizer'
    'loss'
    'metrics'
    'lr_scheduler'
    'trainer'
    'logger'
