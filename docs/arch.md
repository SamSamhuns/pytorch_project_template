# How the repository fits together

This page shows how the repository is laid out, how a training run flows, and how a checkpoint becomes a served model. [README.md](README.md) explains what changed from the old template, and why.

The diagrams are made with [archify](https://github.com/tt-a1i/archify) from the JSON files in [arch/](arch/), and each of their nodes cites the source lines it describes. [Updating the diagrams](#updating-the-diagrams) explains how to change them.

## Layout

![Repository layout](arch/repo-layout.svg)

`train.py`, `test.py`, `inference.py` and `export.py` are short scripts that call into `src/`. Each gets `--cfg`, `-o` and `--id` from `add_common_args`, loads the config with `load_config`, and creates its own run dir with `make_run_dir`.

A config names what to build by its `type` keys, and each is looked up in one place:

| Config key | Looked up by | In |
| --- | --- | --- |
| `model.type` | `MODELS[...]` | `src/models/__init__.py` |
| `dataset.type` | `DATASETS[...]` | `src/datasets/__init__.py` |
| `dataset.preprocess.type` | `build_transforms`, from `TRANSFORMS[...]` | `src/augmentations/__init__.py` |
| `loss.type` | `getattr(torch.nn, ...)` | `src/trainers/classifier_trainer.py` |
| `optimizer.type` | `getattr(torch.optim, ...)` | `src/trainers/classifier_trainer.py` |
| `lr_scheduler.type` | `build_scheduler`, from `torch.optim.lr_scheduler` | `src/trainers/classifier_trainer.py` |
| `metrics.val`, `metrics.test` | `METRICS[...]` and `PLOTS[...]` | `src/metrics/` |

So any loss, optimizer or scheduler in torch works by naming it in the config. A new model, dataset or transform needs its code, an entry in its dict, and its name in a config. [configs/README.md](../configs/README.md) explains every key.

## A training run

![A training run](arch/training-run.svg)

[How a training run flows](../README.md#how-a-training-run-flows), in the main README, walks through these steps. Each script saves to its own run dir, `save_dir/experiment_name/RUN_ID`:

| Script | Writes |
| --- | --- |
| `train.py` | `config.yaml`, `logs/train.log` and curve plots, `models/last.pth` and `models/best.pth`, `tf_logs/` |
| `test.py` | `config.yaml`, `logs/test.log` and curve plots, `tf_logs/` |
| `inference.py` | `config.yaml`, `logs/inference.log`, `logs/pred.txt` |
| `export.py` | `config.yaml`, `logs/export.log`, `models/model.onnx` |

The curve plots are the ROC, PR and calibration curves that `metrics.val` and `metrics.test` name, for binary models only. `tf_logs/` holds the TensorBoard logs, when `trainer.use_tensorboard` is on.

A checkpoint holds `model`, `optimizer`, `scheduler`, `epoch`, `best_val_acc` and `classes`. `train.py -r` restores all but `classes`, which come from the dataset. `test.py` reads only `model`, while `inference.py` and `export.py` also read `classes`.

## From checkpoint to served predictions

![From checkpoint to served predictions](arch/checkpoint-to-server.svg)

- `test.py`, `inference.py` and `export.py` take the config the model was trained with, and build the model and the eval transform from it. Only `test.py` builds the datasets and the trainer.
- `model.onnx` takes any batch size. Its metadata holds the class names under `classes`, and the eval transform's settings under `preprocess`: the image mode, the resize and crop sizes, the mean, the std and the interpolation.
- `server/app.py` loads `model.onnx` with ONNX Runtime. Its image has no torch, so it repeats the eval transform with PIL and numpy from those settings; `tests/test_server.py` checks that its probabilities equal PyTorch's.

## Updating the diagrams

Each JSON file in [arch/](arch/) is the source of the SVG next to it, and cites source lines at the commit in its `meta.repository.revision`. After a change that the diagrams show, update the JSON and its revision. Then render it from the repo root with archify, which also checks the diagram and its sources. `ARCHIFY_DIR` is where archify is installed:

```shell
node ARCHIFY_DIR/bin/archify.mjs finalize tree docs/arch/repo-layout.json .archify/repo-layout.html --repo-root . --quality showcase
```

Use `workflow` for `training-run.json` and `dataflow` for `checkpoint-to-server.json`. The HTML in `.archify/` is the interactive diagram, and its Export menu saves the SVG that replaces the one in `arch/`.
