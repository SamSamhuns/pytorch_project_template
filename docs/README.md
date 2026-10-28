# How this template was modernized, and why

This repository started as a 2021-era PyTorch template for training image classifiers. In October 2026 it was reviewed and modernized in seven phases, one branch each, from `modernize/phase-1` to `modernize/phase-7`. This page states what changed from the previous arrangement and why.

The review that planned the work, with every measurement behind it, was kept in `PLAN.md` until this page replaced it. `git log -- PLAN.md` finds it in the history.

## In brief

- **Results that were silently wrong are fixed.** Pretrained backbones were re-initialized before training, `best.pth` was usually the last epoch, binary ROC-AUC used the wrong scores, and every reported loss was about `batch_size` times too small. The fixes come with tests that fail without them.
- **Documented paths that crashed now work:** the WebDataset config, MNIST training and inference, the FastAPI server, and quantized export, which was removed.
- **There is much less code.** `src/` went from 3,693 lines in 44 files to 832 lines in 22, mostly by deleting dead code and by using what torch, torchvision, scikit-learn and the standard library already provide.
- **The stack is current:** Python 3.11 to 3.14, torch 2.14, one `uv.lock`, ruff in CI, and slim Docker images.
- **The tests catch real bugs.** They cover the fixes above, run offline and take about 12 s.

## The principles behind the changes

1. **Readability first.** A newcomer should be able to read `train.py` from top to bottom and follow the data, the model and the training loop in one sitting.
2. **Delete before adding.** Use torch, torchvision, scikit-learn and the standard library before writing custom code.
3. **Explicit over implicit.** No `mode` strings that change what a constructor does. Side effects such as seeding, creating directories and writing files happen visibly in the entry scripts.
4. **One source of truth** for the dependencies (one lockfile), the preprocessing (one eval transform) and the checkpoints (one format).
5. **Fail fast.** Exceptions are no longer swallowed, and config typos and unknown types raise.
6. **Every fix ships with a test.**

## What changed, area by area

### Results that were wrong

| Before | Now | Why |
| --- | --- | --- |
| The trainer re-initialized every model, wiping its ImageNet weights | The trainer keeps the model's weights | Every fine-tune trained from scratch, and `pretrained_weights` had no effect |
| `best.pth` was compared with the previous validation, not the best one, and also saved on epochs without validation. Testing used the last epoch's weights | `best.pth` is saved only when the val accuracy beats the best so far, and `train.py` tests it | `best.pth` was usually just the last epoch |
| Binary ROC-AUC, PR and calibration curves used the confidence of the predicted class | They use P(class 1) | A perfect classifier scored an AUC of 0.548 |
| The epoch loss summed batch means, then divided by the number of samples | Each batch loss is weighted by its batch size | Every logged loss was about `batch_size` times too small |
| `torch.compile` and DataParallel wrapped the model before its weights were saved | `model.compile()` compiles in place, and DataParallel is gone | Their checkpoints had prefixed keys and couldn't be reloaded |
| Inference skipped normalization, and MNIST inference fed RGB images to a grayscale model | One eval transform serves validation, testing, inference, export and the server | Training and inference saw differently preprocessed images |
| Replacing the head swapped the whole classifier block | Only the last `nn.Linear` is replaced | mobilenet_v3, convnext and vgg crashed |
| Each split computed its own class indices | Every split must have the same class dirs, or loading raises | A test split missing a class was silently mislabelled |
| `-o optimizer.args.lr=1e-3` gave the string `"1e-3"` | Overrides are parsed with OmegaConf's grammar | SGD raised a TypeError |

Smaller fixes: the WebDataset config, `num_workers: 0` and a missing `git` binary crashed at startup. The data-prep scripts dropped `.JPG` files, shuffled an unsorted file list, so a seed gave different splits on different machines, and deleted files when given `-r False`.

### Dead code

About 1,400 lines were deleted together with their tests, none of them on the main path:

- leftovers from earlier time-series and GAN projects, including a feature-importance mode that crashed;
- a warmup scheduler that was never registered and crashed on current torch;
- unused visualization and bounding-box helpers, the only user of OpenCV;
- hand-rolled metrics that scikit-learn provides, some of which raised TypeErrors;
- the TorchScript and `torch.ao` quantization exports, which are deprecated, and quantization failed for both shipped models;
- the in-process TensorBoard server, and two stale scripts.

### Dependencies and tooling

| Before | Now | Why |
| --- | --- | --- |
| Poetry, plus three requirements files and a script that generated one of them | `uv` with one `uv.lock`. `--extra cpu` or `--extra cu130` picks the torch build | The four sources had drifted, and `pip install -r requirements/train.txt` broke ONNX export |
| Python ≥3.10 with torch 2.8 | Python 3.11 to 3.14 with torch 2.14 | Python 3.10 reached its end of life in October 2026, and the newest libraries need 3.11 |
| CI installed CUDA torch on CPU runners and didn't lint | CPU wheels, `ruff check` and `ruff format --check`, on Python 3.11 and 3.14 | Each job downloaded 3.9 GB of CUDA wheels; the CPU wheel is 184 MB |
| A ruff config that reported 1,447 violations | A focused rule set that CI enforces | A config nobody can pass is ignored |
| A Docker image on a 7.4 GB PyTorch devel base, carrying a second CUDA stack | `python:3.13-slim` plus the PyPI torch wheels, which bundle CUDA | 3.2 GB instead of about 11 GB compressed, or 0.41 GB with CPU-only torch |

### Training procedure

| Before | Now | Why |
| --- | --- | --- |
| Models ended in `LogSoftmax`, trained with `NLLLoss` | Models return logits, trained with `CrossEntropyLoss`; the classifier configs add label smoothing | Softmax belongs in metrics and inference, and the exported ONNX model returns the logits that serving expects |
| Checkpoints held only the weights, and resuming restarted at epoch 1 | `last.pth` and `best.pth` hold the model, the optimizer, the LR scheduler, the epoch and the best val accuracy | `-r` continues training where it stopped |
| fp16 autocast without a gradient scaler, in training only | bfloat16 autocast in training and evaluation | fp16 without a scaler risks underflowing gradients, and bf16 needs no scaler |
| A `device` plus a `gpu_device` list for DataParallel | One `device`, such as `cuda`, `cuda:1` or `cpu` | PyTorch recommends DDP over DataParallel; DDP can be added when multi-GPU training is needed |
| A ReduceLROnPlateau that could never fire, and a custom warmup that trained epoch 1 at LR 0 | Any scheduler in `torch.optim.lr_scheduler`, by default `CosineAnnealingLR`, with an optional linear warmup | The LR was constant in every shipped config |
| No augmentation, and the ImageNet constants copied by hand | Random crops and flips for training. The eval transform comes from the pretrained weights | Resize, crop and normalization now match each checkpoint, e.g. V2 weights resize to 232, not 256 |
| Seeding inside the config constructor, and `reproducible: true` in every config | `train.py` seeds; `reproducible` defaults to false | Deterministic mode is slower, and it still wasn't fully deterministic |
| `get_logger` added handlers on every call | Each script sets up logging once, and modules use `logging.getLogger(__name__)` | Every log line was written twice or more |

### Data pipeline

| Before | Now | Why |
| --- | --- | --- |
| A 254-line dataset class that re-implemented `ImageFolder`, and a loader class that existed to carve out a val split | `torchvision.datasets.ImageFolder`, `random_split` and plain `DataLoader`s | The built-ins do the same with less code |
| `__getitem__` caught every exception and returned `None`, which the collate function dropped | Errors raise, and `scripts/find_corrupt_images.py` cleans a dataset once | Batches silently shrank, and the bad file was never reported |
| Three `.item()` calls per training step | The loss and correct counts are summed on the device and read once per epoch | Each `.item()` makes the CPU wait for the GPU |
| Workers restarted every epoch, and memory was always pinned | Persistent workers, and pinned memory only when training on CUDA | Restarting cost 0.25 s per loader on Linux and 3.2 s on macOS |
| One class-sorted, re-encoded tar per split | A dir of shuffled shards per split, holding the original image bytes, with an `info.json` of class names and counts | 58 MB instead of 126 MB, about 1,550 instead of 400 images/s with 4 workers, and 17 instead of 3 classes in the first batch |

### Export, inference and serving

| Before | Now | Why |
| --- | --- | --- |
| Four export modes behind a strategy class, two of them TorchScript | One `export_onnx()`: the dynamo exporter with a dynamic batch size, checked against ONNX Runtime | TorchScript is deprecated, and the other ONNX mode ran the dynamo exporter anyway |
| `export.py` and `inference.py` built the whole trainer, datasets included | They build the model, load the checkpoint and apply the eval transform | Export takes 2.9 s instead of 3.8 s for MNIST, and 6.2 s instead of 8.2 s for MobileNetV2 |
| Inference printed class indices | Checkpoints hold the class names, and inference prints names with probabilities | A prediction should be readable without a lookup table |
| A FastAPI server that couldn't start, in an image without torch | `server/app.py`, a 50-line FastAPI app on ONNX Runtime with one `POST /predict` | ONNX Runtime ran MobileNetV2 4 to 6 times faster than eager PyTorch on CPU |
| The server ran images through the trainer's inference path, which skipped normalization | `model.onnx` stores the class names and the eval transform's settings, which the server applies with PIL and numpy | The server image has no torchvision, and a test checks that the server's probabilities equal PyTorch's |
| A server image with torch and every training dependency | An image with only the `server` dependency group | 113 MB instead of 409 MB compressed |

### Structure

| Before | Now | Why |
| --- | --- | --- |
| `_BaseTrainer`, `BaseTrainer` and `ClassifierTrainer`, whose constructors built the datasets and returned early in some modes | One `ClassifierTrainer` with `fit`, `validate`, `evaluate`, `save` and `load`. `train.py` builds the data and the model | One trainer needs one class, and the flow should read from top to bottom |
| `CustomDictConfig`, a `DictConfig` subclass whose constructor called git, created the run dirs and wrote `config.yaml` | `load_config()` and `make_run_dir()`. The scripts do the side effects | Side effects should be visible, and struct mode makes a typo in a key raise |
| Eight registries, each a dict plus an `init_*` function with a try/except | The loss, optimizer and scheduler come from torch by the name in the config. Models, datasets and transforms are plain dicts | Every torch loss, optimizer and scheduler now works from YAML |
| `--mode` strings, `--dev` and `--gpu_device`, and about 45 lines of argparse code repeated in four scripts | `add_common_args()`, `-o device=cpu`, and `train.py` always tests `best.pth` when there is a test split | `-o` already did what `--dev` did, and modes changed control flow inside constructors |
| Preprocess classes, a `BaseModel`, and a `ClassifierModel` that wrapped a torchvision model | Transform functions, plain `nn.Module` models and `torchvision_classifier()` | Fewer layers to read through |
| `calc_metric` guessed binary or macro averaging from the classes in the split | `metrics.average` in the config | A val split that lacked a class switched to binary averaging silently |
| Tests downloaded MNIST and the MobileNetV2 weights and wrote to `/tmp/pytest` | Offline tests with `FakeData`, a tiny model and pytest's `tmp_path` | About 12 s instead of 53 s, with no network |

## Decisions

| Question | Decision | Why |
| --- | --- | --- |
| Poetry or uv | uv | One lockfile, CPU and CUDA builds as extras, and faster CI and Docker builds |
| Keep WebDataset | Yes, fixed | Streaming shards suits large datasets on network or slow storage |
| Keep the server | Yes, rebuilt on ONNX Runtime | Serving the exported model is part of the template's purpose |
| Load old checkpoints | No | The classifier's keys and the checkpoint format changed; a key map would be code to maintain for a template |
| Multi-GPU | One device for now | Add DDP with `torchrun` when it's needed |
| Flatten `src/` into a few modules | Not now | The other changes delivered most of the value, and the package layout is familiar |

## Considered and not adopted

- **Lightning, Accelerate, Hydra or timm** as core dependencies: the training loop is about 100 lines, and these add more concepts than they remove.
- **W&B or MLflow:** TensorBoard ships with torch, while external trackers add accounts and servers.
- **`torchvision.transforms.v2`:** it measured slower on CPU.
- **GPU JPEG decoding, DALI or FFCV:** too much machinery for a template.
- **`channels_last`, CUDA graphs or fused optimizers:** small or fragile gains for these CNNs.
- **Server batching or async workers:** sync endpoints in FastAPI's thread pool are enough.
- **torchao quantization:** a new dependency, when ONNX Runtime's `quantize_static` would work with an existing one.
- **EMA, MixUp, CutMix, gradient clipping or early stopping:** add them per project.
- **Config inheritance:** four small YAML files don't need it.

## Upgrading from the old template

| Old | New |
| --- | --- |
| `poetry install`, or `pip install -r requirements/train.txt` | `uv sync --extra cpu`, or `uv sync --extra cu130` for NVIDIA GPUs |
| `python train.py` with a default config | `python train.py --cfg CONFIG`; the config is required |
| `test.py`, `export.py` or `inference.py` with `-r CHECKPOINT` | `-w CHECKPOINT`. `train.py` keeps `-r` for resuming |
| `--dev cpu`, `--gpu_device 0` | `-o device=cpu`, `-o device=cuda:0` |
| `--mode TRAIN` | `train.py` tests `best.pth` when the dataset has a test split; `-o dataset.args.test_path=null` skips it |
| `export.py --mode ONNX_TS` or `ONNX_DYNAMO` | `export.py --cfg CONFIG -w CHECKPOINT` writes `model.onnx` |

- **Config types:** `model.type: ClassifierModel` is now `torchvision_classifier`, `dataset.preprocess` takes one `type`, `imagenet_transforms` or `mnist_transforms`, with its `args`, and the loss is `CrossEntropyLoss`, as the models return logits.
- **Config sections:** `dataloader` holds only `batch_size` and `num_workers`, without `type` and `args`, `validation_split` moved to `dataset.args`, and `metrics.average` is new.
- **Removed config keys:** `mode`, `git_hash`, `gpu_device`, `quant_backend`, `trainer.type`, `trainer.resume_checkpoint`, `trainer.save_best_only`, `trainer.weight_save_freq`, `trainer.verbosity`, `trainer.tensorboard_port`, `dataset.num_classes` and `model.info`. Unknown keys now raise, so compare a copied config with [configs/README.md](../configs/README.md).
- **Checkpoints** from before the modernization don't load.
- **WebDataset shards** written by the old converter need converting again with `scripts/convert_dataset_to_tar.py`.

## Before and after, measured

| | Before | Now |
| --- | --- | --- |
| `src/` | 3,693 lines in 44 files | 832 lines in 22 files |
| Tests | 119 passed, catching none of the bugs above | 67 passed and 3 that need CUDA, offline, in about 12 s |
| CI download per job | 3.9 GB | 184 MB |
| Training image, compressed | about 11 GB | 3.2 GB with CUDA, 0.41 GB with CPU-only torch |
| Server | couldn't start | 113 MB image, ONNX Runtime |

Everything was measured on CPU. GPU-only behaviour, such as bfloat16 autocast and pinned memory, wasn't verified, as the machine's GPUs were busy.
