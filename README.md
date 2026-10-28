# Pytorch Project Template, Computer Vision

[![tests](https://github.com/SamSamhuns/pytorch_project_template/actions/workflows/main.yml/badge.svg)](https://github.com/SamSamhuns/pytorch_project_template/actions/workflows/main.yml)[![Codacy Badge](https://app.codacy.com/project/badge/Grade/8d13d18c6af947329b09ed473231d36d)](https://www.codacy.com/gh/SamSamhuns/pytorch_project_template/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=SamSamhuns/pytorch_project_template&amp;utm_campaign=Badge_Grade)

[![Python 3.11](https://img.shields.io/badge/python-3.11-green.svg)](https://www.python.org/downloads/release/python-3110/)[![Python 3.12](https://img.shields.io/badge/python-3.12-green.svg)](https://www.python.org/downloads/release/python-3120/)[![Python 3.13](https://img.shields.io/badge/python-3.13-green.svg)](https://www.python.org/downloads/release/python-3130/)[![Python 3.14](https://img.shields.io/badge/python-3.14-green.svg)](https://www.python.org/downloads/release/python-3140/)

This is a template for a PyTorch Project for training, testing, inference demo, and FastAPI serving along with Docker support.

- [Pytorch Project Template, Computer Vision](#pytorch-project-template-computer-vision)
  - [Setup](#setup)
  - [Train](#train)
  - [Custom Training](#custom-training)
    - [Data Preparation](#data-preparation)
      - [Example Training: Image Classification](#example-training-image-classification)
    - [WebDataset for large scale training](#webdataset-for-large-scale-training)
  - [Test](#test)
  - [Export](#export)
  - [TensorBoard logging](#tensorboard-logging)
  - [Inference](#inference)
  - [Docker](#docker)
    - [Training and testing](#training-and-testing)
    - [For serving the model with FastAPI](#for-serving-the-model-with-fastapi)
  - [Utility functions](#utility-functions)
  - [Profiling PyTorch](#profiling-pytorch)
  - [Acknowledgements](#acknowledgements)

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the environment with the PyTorch build for your machine:

```shell
uv sync --extra cpu    # CPU-only PyTorch (Linux, Windows and macOS)
uv sync --extra cu130  # PyTorch with CUDA 13.0 for NVIDIA GPUs (driver >= 580)
```

Pass the same `--extra` every time you run `uv sync`; without one, uv removes PyTorch. Run commands with `uv run`, e.g. `uv run python train.py --cfg configs/mnist_config.yaml`, or activate the environment with `source .venv/bin/activate`.

-   This also installs the FastAPI server's dependencies, the `server` group, which the default `dev` group includes.
-   For a pip requirements file: `uv export --extra cpu --no-hashes > requirements.txt`

## Train

Example training for mnist digit classification:

```shell
python train.py --cfg configs/mnist_config.yaml

# resume a run from its last checkpoint: weights, optimizer, LR schedule and epoch
python train.py --cfg configs/mnist_config.yaml -r checkpoints/mnist_image_classifier/RUN_ID/models/last.pth
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
# list the images PIL can't read, which would stop training; add -r to remove them
python scripts/find_corrupt_images.py -s data/SOURCE_DATASET

# generate an id to name classmap
python scripts/generate_classmap_from_dataset.py --sd data/SOURCE_DATASET --mp data/ID_2_CLASSNAME_MAP_TXT_FILE

# create train val test split, also creates an index to classname mapping txt file
python scripts/train_val_test_split.py --rd data/SOURCE_DATASET --td data/SOURCE_DATASET_SPLIT --vs VAL_SPLIT_FRAC -ts TEST_SPLIT_FRAC

# OPTIONAL duplicate train data if necessary
python scripts/duplicate_data.py --rd data/SOURCE_DATASET_SPLIT/train --td data/SOURCE_DATASET_SPLIT/train -n TARGET_NUMBER

# create a custom config file based on configs/classifier_cpu_config.yaml and modify train parameters
cp configs/classifier_cpu_config.yaml configs/custom_classifier_cpu_config.yaml
```

#### Example Training: Image Classification

Sample data used in the custom image classification training downloaded from <https://www.kaggle.com/datasets/umairshahpirzada/birds-20-species-image-classification>.

```shell
# train on custom data with custom config
python train.py --cfg custom_classifier_cpu_config.yaml
```

### WebDataset for large scale training

WebDataset streams the images from tar shards, which suits large datasets on network or slow storage better than many small files. Convert each split of a dataset in the structure above to its own dir of shards. The images are stored as they are, without re-encoding, and an `info.json` next to the shards holds the class names and the image count.

```shell
# writes e.g. data/birds_webdataset/train/000000.tar, 000001.tar, ... and info.json
for split in train valid test; do
  python scripts/convert_dataset_to_tar.py --sd data/birds_dataset/$split --td data/birds_webdataset/$split
done
```

Each DataLoader worker reads its own shards, so write at least as many shards as `dataloader.num_workers`. `--maxcount` sets the images per shard (default 1000).

An example configuration for training with the WebDataset format is provided in `configs/classifier_webdataset_cpu_config.yaml`.

```shell
# example training with webdataset tar data format
python train.py --cfg configs/classifier_webdataset_cpu_config.yaml
```

## Test

Test based on CONFIG_FILE. By default testing is done for mnist classification.

```shell
python test.py --cfg CONFIG_FILE
```

## Export

```shell
python export.py --cfg CONFIG_FILE -r checkpoints/EXPERIMENT_NAME/RUN_ID/models/best.pth
```

This writes `model.onnx` to the export run's `models` dir, e.g. `checkpoints/EXPERIMENT_NAME/export_DATE_TIME/models/model.onnx`, and checks that ONNX Runtime's outputs match the PyTorch model's. The ONNX model takes any batch size, and holds the class names and the settings of the eval transform, which the server reads.

## TensorBoard logging

TensorBoard logs are saved in each run's `tf_logs` dir, `SAVE_DIR/EXPERIMENT_NAME/RUN_ID/tf_logs`. Logs include the config, train/val epoch accuracy/loss, the train loss every `batch_log_freq` batches, and the last preprocessed train batch per epoch.

To start a tensorboard server reading logs from the `experiment` dir exposed on port localhost port `6007`:

```shell
tensorboard --logdir=TF_LOG_DIR --port=6006
```

## Inference

```shell
python inference.py --cfg CONFIG_FILE -r checkpoints/EXPERIMENT_NAME/RUN_ID/models/best.pth -s IMAGE_FILE_OR_DIR
```

This logs the most likely class of each image with its probability, and writes them to the run's `logs/pred.txt`.

## Docker

Install docker in the system first:

### Training and testing

```shell
bash scripts/build_docker.sh  # builds the CUDA image; add --build-arg TORCH_EXTRA=cpu for a CPU-only image
bash scripts/run_docker.sh    # runs the image with a shared checkpoints_docker volume outside the container
# inside the docker container
python train.py
```

The container runs on CPU unless you pass GPUs with `-g`, e.g. `bash scripts/run_docker.sh -g all` or `-g device=0`. GPUs need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and an NVIDIA driver >= 580 on the host.

### For serving the model with FastAPI

The server runs a `model.onnx` from `export.py` with ONNX Runtime, so its image has no PyTorch.

```shell
bash server/build_server_docker.sh
mkdir -p checkpoints_server && cp checkpoints/EXPERIMENT_NAME/export_DATE_TIME/models/model.onnx checkpoints_server/
bash server/run_server_docker.sh -p 8080
# the 5 most likely classes of an image, with their probabilities
curl -F file=@IMAGE_FILE "localhost:8080/predict?top_k=5"
```

Without Docker, run `MODEL_PATH=path/to/model.onnx uvicorn server.app:app --port 8080` from the repo root.

## Utility functions

Clean cached builds, pycache, .DS_Store files, etc:

```shell
bash scripts/cleanup.sh
```

Count number of files in sub-directories in PATH

```shell
bash scripts/count_files.sh PATH
```

## Profiling PyTorch

-   Line by line GPU memory usage profiling [pytorch_memlab](https://github.com/Stonesjtu/pytorch_memlab)
-   Line by line time used profiling [line_profiler](https://github.com/pyutils/line_profiler)

## Acknowledgements

-   <https://github.com/victoresque/pytorch-template>
-   WebDataset <https://modelzoo.co/model/webdataset>
-   PyTorch Ecosystem Tools <https://pytorch.org/ecosystem/>
