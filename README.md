# Pytorch Project Template

This is a template for a PyTorch Project for training, testing, inference demo, and FastAPI serving along with Docker support.

## Project Structure

    ├── checkpoints
    ├── configs
    │   ├── base_configs.json
    ├── copy_project.py
    ├── inference.py
    ├── logs
    ├── modules
    │   ├── agents
    │   │   ├── base_agent.py
    │   ├── dataloaders
    │   │   └── base_dataloader.py
    │   ├── datasets
    │   │   ├── base_dataset.py
    │   ├── loggers
    │   │   └── base_logger.py
    │   ├── losses
    │   │   ├── bce.py
    │   ├── models
    │   │   ├── example_model.py
    │   ├── optimizers
    │   │   └── __init__.py
    │   ├── schedulers
    │   │   └── __init__.py
    │   └── utils
    │       ├── util.py
    ├── requirements
    │   ├── minimal.txt
    ├── requirements.txt
    ├── server.py
    ├── test.py
    ├── train.py

## Setup

Use python venv or a conda env to install requirements:

-   Install full-requirements: `pip install -r requirements.txt`
-   Install train/test/minimal requirements: `pip install -r requirements/train|test|minimal.txt`

## Train

## Test

## Inference demo

## Dockerizing

### Utility functions

To cleanup:

`bash modules/utils/cleanup.sh`

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
