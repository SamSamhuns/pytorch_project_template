# Pytorch Project Template, Computer Vision

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

-   Install full-requirements: `pip install -r requirements.txt`
-   Install train/minimal requirements: `pip install -r requirements/train|minimal.txt`

## Train

Example training for mnist digit classification with default config:

```shell
$ python train.py
```

## Custom Training

### Image Classification

set training data inside `data` directory in the following format:

    data
    |── CUSTOM_DATASET
        ├── CLASS 1
        |   ├── img1
        |   └── img2
        |   ├── ...
        ├── CLASS 2
        |   ├── img1
        |   └── img2
        |   ├── ...

```shell
# create train val test split
$ python modules/utils/train_val_test_split.py -rd data/CUSTOM_DATASET -td data/CUSTOM_DATASET_SPLIT -vs VAL_SPLIT_FRAC -ts TEST_SPLIT_FRAC
# OPTIONAL duplicate train data if necessary
$ python modules/utils/duplicate_data.py -rd data/CUSTOM_DATASET_SPLIT/train -td data/CUSTOM_DATASET_SPLIT/train -n TARGET_NUMBER
# create a custom config file based on configs/classifier_config.py and modify train parameters
$ cp configs/classifier_config.py configs/custom_classifier_config.py
# train on custom data with custom config
$ python train.py -c custom_classifier_config.py
```

## Test

Test based on CONFIG_FILE. By default testing is done for mnist classification.

```shell
$ python test.py -c CONFIG_FILE
```

## Tensorboard logging

All tensorboard logs are saved in the `TENSORBOARD_EXPERIMENT_DIR` setting in the config file. Logs include train/val epoch accuracy/loss, graph, and preprocessed images per epoch.

To start a tensorboard server reading logs from the `experiment` dir exposed on port localhost port `6007`:

```shell
$ tensorboard --logdir=experiments --port=6006
```

## Inference

## Docker

Install docker in the system first:

### For training and testing

```shell
$ bash scripts/build_docker.sh  # builds the docker image
$ bash scripts/run_docker.sh    # runs the previous docker image creating a shared volume checkpoint_docker outside the container
# inside the docker container
$ python train.py
```

Using gpus inside docker for training/testing:

`--gpus device=0,1 or all`

### For serving the model with FastAPI

```shell
$ bash server/build_server_docker.sh
$ bash server/run_server_docker.sh -h/--http 8080
```

### Utility functions

To cleanup cached builds, pycache, .DS_Store files, etc:

    bash scripts/cleanup.sh

To copy project structure:

    $ python3 copy_project.py ../NewProject

## PyTorch Ecosystem Tools

These [Ecosystem Tools](https://pytorch.org/ecosystem/) add to the base PyTorch Ecosystem.

## Example Project

### Bird Recognition task

### Landmark recognition task

Model reached LB score 0.111 using a single ResNet50 for the 2018 Google Landmark Recognition Kaggle Challenge

Experimental configuration of model training & testing:

    Architecture:
      ResNet50

    Input size:
      224x224

    Data augmentation:
      Resize the original images to 256x256
      Crop at random position
      Randomly horizontally flip it
      Demean and normalize it

    Batch size:
      32

    Initial weights:
      Pretrained on ImageNet

    Initial learning rate:
      1e-4

    Learning rate decay:
      Learning rate is halved at Epoch 5 and halved again at Epoch 7

    Max Epochs:
      8

    validation:
      1/8 training images
      At test stage, center-crop is used instead of random-crop

### Acknowledgements

-   <https://github.com/victoresque/pytorch-template>
-   WebDataset <https://modelzoo.co/model/webdataset>
