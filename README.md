# Pytorch Project Template, Computer Vision

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/8d13d18c6af947329b09ed473231d36d)](https://www.codacy.com/gh/SamSamhuns/pytorch_project_template/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=SamSamhuns/pytorch_project_template&amp;utm_campaign=Badge_Grade)

[![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)](https://www.python.org/downloads/release/python-380/)[![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)](https://www.python.org/downloads/release/python-390/)[![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/downloads/release/python-3100/)

This is a template for a PyTorch Project for training, testing, inference demo, and FastAPI serving along with Docker support.

## Project Structure

    ├── checkpoints
    ├── configs
    ├── modules
    │   ├── agents
    |   |── augmentations
    │   ├── dataloaders
    │   ├── datasets
    │   ├── loggers
    │   ├── losses
    │   ├── models
    │   ├── optimizers
    │   ├── schedulers
    │   └── utils
    ├── server
    |-- requirements
    ├── requirements.txt
    ├── copy_project.py
    ├── inference.py
    ├── server.py
    ├── test.py
    ├── train.py

## Setup

Use `python venv` or a `conda env` to install requirements:

-   Install full train requirements: `pip install -r requirements/train.txt`
-   Install minimal inference requirements: `pip install -r requirements/inference.txt`

## Train

Example training for mnist digit classification:

```shell
python train.py --cfg configs/mnist_config.json
```

## Custom Training

### Data Preparation

Set training data inside `data` directory in the following format:

    data
    |── SOURCE_DATASET
        ├── CLASS 1
        |   ├── img1
        |   └── img2
        |   ├── ...
        ├── CLASS 2
        |   ├── img1
        |   └── img2
        |   ├── ...

    Note: ImageNet style class_dir->subdirs->subdirs->images... is also supported

```shell
# generate an id to name classmap
python modules/utils/generate_classmap_from_dataset.py --sd data/SOURCE_DATASET --mp data/ID_2_CLASSNAME_MAP_TXT_FILE

# create train val test split, also creates an index to classname mapping txt file
python modules/utils/train_val_test_split.py --rd data/SOURCE_DATASET --td data/SOURCE_DATASET_SPLIT --vs VAL_SPLIT_FRAC -ts TEST_SPLIT_FRAC

# OPTIONAL duplicate train data if necessary
python modules/utils/duplicate_data.py --rd data/SOURCE_DATASET_SPLIT/train --td data/SOURCE_DATASET_SPLIT/train -n TARGET_NUMBER

# create a custom config file based on configs/classifier_cpu_config.json and modify train parameters
cp configs/classifier_cpu_config.json configs/custom_classifier_cpu_config.json
```

#### Example Training: Image Classification

Sample data used in the custom image classification training downloaded from <https://www.kaggle.com/datasets/gpiosenka/100-bird-species>.

```shell
# train on custom data with custom config
python train.py --cfg custom_classifier_cpu_config.json
```

### WebDataset for large scale training

Convert existing dataset to a `tar` archive format used by WebDataset. The data directory must match the structure above.

```shell
# ID_2_CLASSNAME_MAP_TXT_FILE is generated using the modules/utils/train_val_test_split.py file
# convert train/val/test splits into tar archives
python modules/utils/convert_dataset_to_tar.py --sd data/SOURCE_DATA_SPLIT --td data/TARGET_TAR_SPLIT.tar --mp ID_2_CLASSNAME_MAP_TXT_FILE
```

An example configuration for training with the WebDataset format is provided in `configs/classifier_webdataset_cpu_config.json`.

```shell
# example training with webdataset tar data format
python train.py --cfg configs/classifier_webdataset_cpu_config.json
```

## Test

Test based on CONFIG_FILE. By default testing is done for mnist classification.

```shell
python test.py --cfg CONFIG_FILE
```

## Tensorboard logging

All tensorboard logs are saved in the `tensorboard_log_dir` setting in the config file. Logs include train/val epoch accuracy/loss, graph, and preprocessed images per epoch.

To start a tensorboard server reading logs from the `experiment` dir exposed on port localhost port `6007`:

```shell
tensorboard --logdir=TF_LOG_DIR --port=6006
```

## Inference

## Docker

Install docker in the system first:

### For training and testing

```shell
bash scripts/build_docker.sh  # builds the docker image
bash scripts/run_docker.sh    # runs the previous docker image creating a shared volume checkpoint_docker outside the container
# inside the docker container
python train.py
```

Using gpus inside docker for training/testing:

`--gpus device=0,1 or all`

### For serving the model with FastAPI

```shell
bash server/build_server_docker.sh -m pytorch/onnx
bash server/run_server_docker.sh -h/--http 8080
```

### Utility functions

Clean cached builds, pycache, .DS_Store files, etc:

```shell
bash scripts/cleanup.sh
```

Copy project structure:

```shell
python copy_project.py ../NewProject
```

Count number of files in sub-directories in PATH

```shell
bash scripts/count_files.sh PATH
```

### Profiling PyTorch

-   Line by line GPU memory usage profiling [pytorch_memlab](https://github.com/Stonesjtu/pytorch_memlab)
-   Line by line time used profiliing [line_profiler](https://github.com/pyutils/line_profiler)

### Acknowledgements

-   <https://github.com/victoresque/pytorch-template>
-   WebDataset <https://modelzoo.co/model/webdataset>
-   PyTorch Ecosystem Tools <https://pytorch.org/ecosystem/>
