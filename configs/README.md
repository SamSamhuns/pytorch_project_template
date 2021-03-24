# Required Keys for Configuration JSON file

An example Configuration file is provided at `configs/base_configs.json`

## Required top level Keys

    'NAME'
    "SEED"
    'USE_CUDA'
    'N_GPU'
    'GPU_DEVICE'
    'ARCH'
    'DATASET'
    'DATALOADER'
    'OPTIMIZER'
    'LOSS'
    'METRICS'
    'LR_SCHEDULER'
    'TRAINER'
    'LOGGER'

## Required sub-level Keys

    'ARCH' = ['TYPE', 'PRETRAINED', 'INPUT_WIDTH', 'INPUT_HEIGHT']

    DATASET = ['RAW_DATA_ROOT_DIR', 'PROC_DATA_ROOT_DIR', 'TRAIN_DIR', 'TEST_DIR', 'NUM_CLASSES']

    DATALOADER = ['TYPE', 'BATCH_SIZE', 'SHUFFLE']

    OPTIMIZER = ['TYPE', 'LR']

    LR_SCHEDULER = ['TYPE']

    TRAINER = ['RESUME', 'EPOCHS', 'CHECKPOINT_DIR']

    LOGGER = ['DIR']
