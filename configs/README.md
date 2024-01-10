# Required Keys for Configuration JSON file

An example Configuration file is provided at `classifier_cpu_config.json`

## Required top level Keys

    'name'
    "seed"
    'cudnn_deterministic' # for repeating results together with seed, WARNING: GPU time might be slower than CPU if true
    'cudnn_benchmark'     # set to true for faster training with gpu, as it usues faster funcs on subsequent runs
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
