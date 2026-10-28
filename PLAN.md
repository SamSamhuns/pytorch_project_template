# Modernization plan: pytorch_project_template

**Review date:** 2026-10-27
**Baseline:** `master` at `25258e5` (torch 2.8.0, Python ≥3.10, Poetry). File:line links refer to this baseline, so they drift as phases land.

**How this was produced:**

- Two independent reviewers:
  - correctness and dependencies
  - speed and training procedures
- One readability and design pass.

**What the tags mean:**

- ✅ means the claim was reproduced on CPU. Runs used an exact copy of `poetry.lock` and, where relevant, the latest releases (torch 2.14.1, Python 3.13 and 3.14).
- 🔎 means reasoned but not reproduced here, mostly GPU-only behavior.
- Effort is **S** (under 1 hour), **M** (about half a day) or **L** (1–2 days).
- Each item is tagged **R+** (code gets simpler), **R0** (neutral) or **R−** (adds code, only where the value clearly justifies it).

This repo puts human readability first. The plan therefore deletes code and uses torch, torchvision, scikit-learn and stdlib built-ins rather than adding abstractions. Nitpicks are kept out of the phases and listed separately in [Nitpicks](#nitpicks-segregated).

---

## Summary

- **The main training path runs, and the test suite passes (119 tests)**, but the tests catch none of the bugs below.
- **Some bugs silently produce wrong results:**
  - Pretrained weights are re-initialized before training, so every "fine-tune" trains from scratch.
  - `best.pth` is usually just the last epoch.
  - Binary ROC-AUC and PR curves are computed from the wrong scores. A perfect classifier scores an AUC of 0.548.
  - Every reported loss is about `batch_size` times too small.
- **Several documented paths are broken:**
  - The WebDataset config crashes at startup.
  - The FastAPI server can't start, and its Docker image has no torch.
  - MNIST inference crashes.
  - Quantized export fails for both shipped models.
  - Several backbones named in the model's docstring crash.
- **About 1,400 lines can be deleted outright** (Phase 2). They are dead code, leftovers from an earlier time-series and GAN project, or hand-rolled copies of built-ins. After Phases 5–7, `src/` should shrink from 3,693 lines to roughly 1,500–1,800 (est.).
- **The stack is out of date:**
  - torch is six minor versions behind (2.8 → 2.14.1).
  - Python 3.10 reached end-of-life in October 2026.
  - TorchScript and `torch.ao.quantization` are deprecated.
  - CI downloads about 3.9 GB of CUDA wheels per matrix leg on CPU-only runners.

### Top 10 changes

| # | Change | Why | Effort | Item |
| --- | --- | --- | --- | --- |
| 1 | Stop re-initializing pretrained backbones | Every fine-tune currently trains from scratch | S | 1.1 |
| 2 | Fix best-checkpoint selection and test on `best.pth` | `best.pth` is usually the last epoch | S | 1.2 |
| 3 | Pass P(class 1) to ROC-AUC, PR and calibration | A perfect classifier gets AUC 0.548 | S | 1.3 |
| 4 | Delete dead and broken code | About 1,400 fewer lines to read | M | Phase 2 |
| 5 | Move to Python ≥3.11 and torch 2.14 with one lockfile | EOL Python, deprecated APIs, broken pip installs | M | 3.1–3.3 |
| 6 | Use CPU wheels and run lint in CI; slim the Docker base image | CI download drops from 3.9 GB to 184 MB per leg; the image drops from ~11 GB to ~4 GB (est.) | S–M | 3.4–3.6 |
| 7 | Save full-state checkpoints for real resume; use `model.compile()` | Resume restarts the optimizer and schedule; compiled checkpoints don't reload | S | 1.5, 4.2 |
| 8 | Add real training augmentation; use one eval transform everywhere | There is no augmentation today, and preprocessing differs between train and inference | S | 1.6, 4.6 |
| 9 | Use one ONNX (dynamo) export path and an ONNX Runtime server | TorchScript is deprecated; the server can't start | M | 6.1–6.4 |
| 10 | Collapse the config, trainer and registry layers | 3 trainer classes, 8 copy-pasted registries, and a config object with side effects | L | Phase 7 |

### Design principles behind the plan

1. **Readability first.** A newcomer should be able to trace `train.py` → data → model → `Trainer.fit` in one sitting.
2. **Delete before adding.** Use built-ins (torch, torchvision, sklearn, stdlib) before custom code.
3. **Explicit over implicit.**
   - No `mode` strings that change what a constructor does.
   - Side effects (seeding, creating directories, writing files) happen visibly in the entry scripts.
4. **One source of truth** for each of these:
   - dependencies: one lockfile
   - preprocessing: one eval transform
   - checkpoints: one format
5. **Fail fast.** Don't swallow exceptions. Config typos and missing classes should raise.
6. **Every fix ships with a test.** Each bug fix includes the smallest test that fails without it. Unit tests run offline in seconds.

---

## Phase 1: Correctness (silently wrong results and crashes)

Each fix should land with one small regression test. None of these bugs is caught by the current suite.

**Status (2026-10-27): done.**

- **Tests:** 127 passed, 27 skipped (CUDA-only), on both the exact lock (torch 2.8, Python 3.12) and the latest stack (torch 2.14.1, Python 3.13). Each new regression test fails on `25258e5`.
- **Smoke runs passed:**
  - MNIST: train → `best.pth` → `test.py` → `inference.py` on RGB `.PNG` files → ONNX export
  - a binary classifier with `torch_compile_model: true` and the ROC/PR/calibration plots
  - the WebDataset config
- **Changes beyond the item list:**
  - DataParallel was removed as part of 1.5, so multi-GPU lists now raise. This pulls 4.4 forward.
  - The `mode`, `save_best_only`, `weight_save_freq` and `prefetch_factor` config keys were removed.
    - Copied configs that still set `prefetch_factor` fail at startup with a TypeError.
    - Stale `save_best_only` and `weight_save_freq` keys are silently ignored.
  - `torch_compile_model` now compiles at the start of `train()`, so test, export and inference never compile. This also fixes export with compile on: 3 of 4 modes failed before.
  - `-r` must name a checkpoint file.
    - Passing a directory used to load its newest `.pth`, which is now `last.pth` rather than `best.pth`.
    - `find_latest_file_in_dir` was deleted.
  - `feat_extract` returns the input of the last Linear layer, e.g. 1024 features instead of 576 for mobilenet_v3 and 4096 instead of 25,088 for vgg11.
  - The broken `main()` demo in classifier_model.py was deleted with the rewrite. This pulls one Phase 2 item forward.
- **Independent review:** found two regressions in the first version, both now fixed and verified:
  - TorchScript testing failed with compile on.
  - `-r <dir>` picked `last.pth`.
- **Not covered by a test:** the 1.11 script fixes. They are one-line changes in standalone scripts.

### 1.1 Pretrained backbone weights are wiped before training (critical, ✅, S, R+)

- **Where:** [classifier_trainer.py:88-100](src/trainers/classifier_trainer.py#L88-L100) calls [base_model.py:18-52](src/models/base_model.py#L18-L52).
- **Problem:** `initialize_weights("he")` runs on every model, including torchvision backbones loaded with ImageNet weights.
  - After `ClassifierTrainer(...)`, 0 of 156 backbone tensors are still pretrained.
  - BatchNorm affine parameters are reset while the running stats stay ImageNet, which is an inconsistent state.
  - The `pretrained_weights` config key is effectively ignored.
- **Fix:**
  - Delete the block, keeping the `get_model_params` log line.
  - Models that want a custom init do it in their own `__init__`.
  - Then delete `_he_init`, `_glorot_init` and `initialize_weights`.
- **Test:** after trainer init, a backbone tensor still equals the downloaded weight.

### 1.2 `best.pth` is not the best model, and TRAIN_TEST tests the last epoch (critical, ✅, S, R+)

- **Where:** [classifier_trainer.py:257-267](src/trainers/classifier_trainer.py#L257-L267), [classifier_trainer.py:292-298](src/trainers/classifier_trainer.py#L292-L298), [train.py:62-67](train.py#L62-L67)
- **Problem:**
  - It compares against the *previous* validation, not the best so far.
  - It runs on epochs that had no validation.
  - Any val metric other than accuracy or loss has an empty history. The gpu, mnist and webdataset configs include `f1_score`, so `best.pth` is overwritten every epoch.
  - Reproduced with val accuracy 90 → 50 → 60: `best.pth` ends up holding the epoch with 60.
  - `test()` then evaluates the in-memory last-epoch weights.
  - With `save_best_only: false`, the final epochs are never saved when `epochs % weight_save_freq != 0`.
- **Fix:**
  - Track `self.best_val_acc` and save `best.pth` inside `validate()` only when it improves.
  - Save `last.pth` every epoch (see 4.2).
  - `train.py` loads `best.pth` before `test()`.
  - Delete `best_val_metric_dict` and the save loop.
- **Test:** stubbed val accuracies 90/50/60 leave `best.pth` holding the epoch with 90.

### 1.3 Binary ROC-AUC, ROC, PR and calibration use the wrong scores (critical, ✅, S, R+)

- **Where:** [classifier_trainer.py:221-226](src/trainers/classifier_trainer.py#L221-L226), [metrics/\_\_init\_\_.py:59-73](src/metrics/__init__.py#L59-L73)
- **Problem:**
  - `y_score` is `softmax(...).max(1)`, the confidence of the *predicted* class, not P(class=1).
  - The ROC and PR plots are drawn from hard labels.
  - A perfect classifier gets AUC 0.548. A decent noisy one gets 0.477 instead of its true 0.837.
- **Fix:**
  - `probs = output.softmax(1); y_pred = probs.argmax(1); y_score = probs[:, 1]` for the binary case.
  - The curve plots take `y_score`.
- **Test:** perfect binary predictions give AUC 1.0.

### 1.4 Every reported loss is about `batch_size` times too small (high, ✅, S, R0)

- **Where:** [classifier_trainer.py:147](src/trainers/classifier_trainer.py#L147) with [:174](src/trainers/classifier_trainer.py#L174), [:220](src/trainers/classifier_trainer.py#L220) with [:288](src/trainers/classifier_trainer.py#L288), and [:366](src/trainers/classifier_trainer.py#L366)
- **Problem:**
  - The code sums per-batch *mean* losses, then divides by the number of samples.
  - At batch size 16 it reported 0.00681 when the true mean NLL was 0.06974.
  - Logs and TensorBoard are wrong. ReduceLROnPlateau is unaffected because its threshold is relative.
- **Fix:** accumulate `loss.item() * data.size(0)`, or accumulate a tensor (see 5.2).
- **Test:** the reported epoch loss equals the mean loss over the dataset.

### 1.5 Checkpoints saved under `torch.compile` or DataParallel can't be reloaded (high, ✅, S, R+)

- **Where:** [classifier_trainer.py:79-119](src/trainers/classifier_trainer.py#L79-L119), [base_trainer.py:67-112](src/trainers/base_trainer.py#L67-L112)
- **Problem:**
  - The model is wrapped *before* weights are loaded, and the wrapper's `state_dict` is what gets saved.
  - Keys come out as `_orig_mod.*` or `module.*`.
  - inference.py, export and CPU testing all fail to load them, and a plain checkpoint fails to load into a compiled model.
- **Fix:**
  - Load weights first.
  - Then call `model.compile()`, which compiles in place and keeps clean keys (PyTorch docs: "where to apply compile").
  - Drop DataParallel (4.4) and the key-renaming block.
  - Use a single `torch.load(path, map_location="cpu", weights_only=True)`.
- **Test:** a checkpoint saved with `torch_compile_model: true` loads into a plain model.

### 1.6 Preprocessing differs between train and inference, and MNIST inference crashes (high, ✅, S–M, R+)

- **Where:**
  - [classifier_dataset.py:66-74](src/datasets/classifier_dataset.py#L66-L74) and [:90-97](src/datasets/classifier_dataset.py#L90-L97)
  - [common.py:250-254](src/utils/common.py#L250-L254)
  - [classifier_transforms.py:26-41](src/augmentations/classifier_transforms.py#L26-L41)
  - [mnist_transforms.py:9-11](src/augmentations/mnist_transforms.py#L9-L11)
  - [classifier_trainer.py:419-423](src/trainers/classifier_trainer.py#L419-L423)
- **Problems:**
  - **(a) Normalization is missing at inference.** `CustomImageClassifierPreprocess` computes the dataset mean/std and appends `Normalize` to the train, val and test transforms, but `inference.py` and the server never get it.
  - **(b) State leaks between runs.** `add_tfms` mutates class-level `Compose` objects, so a second dataset in the same process silently reuses the first dataset's statistics.
  - **(c) Mixed aspect ratios can't be batched.** `Resize(224)` without a crop fails with `stack expects each tensor to be equal size`.
  - **(d) MNIST inference crashes.** Images are converted to RGB, giving 3 channels to a 1-channel model.
  - **(e) MNIST training crashes at startup ✅** (found during Phase 1). [base_trainer.py:175-186](src/trainers/base_trainer.py#L175-L186) records the dataset's transforms and normalization stats in the config. `MnistDataset` has no `train_transform`, so this raises `AttributeError`. The documented MNIST smoke run therefore never worked.
- **Fix:**
  - Delete `CustomImageClassifierPreprocess`, the automatic mean/std branches, `add_tfms` and `get_img_dset_mean_std`.
  - Delete the BaseTrainer block that records transforms, mean and std. `config.yaml` is then saved once, by the config class.
  - Normalization comes from the pretrained weights (4.6), or from constants in the config when training from scratch.
  - Add `Grayscale()` to the MNIST eval transform.
  - Use a single eval transform for val, test, inference and export.
- **Test:** the inference transform equals the test transform.

### 1.7 Head replacement crashes for several torchvision backbones (high, ✅, S, R+)

- **Where:** [classifier_model.py:46-77](src/models/classifier_model.py#L46-L77)
- **Problem:** the new `Linear` is sized from the last layer but replaces the *whole* head.
  - `mobilenet_v3_*`, `convnext_tiny` and `vgg11` crash.
  - `squeezenet1_0` raises AttributeError.
  - The docstring lists several of these as supported.
- **Fix:** replace only the last `nn.Linear`. This was verified on mobilenet_v2/v3, resnet18, efficientnet_b0, convnext_tiny, vgg11, vit_b_16, swin_t, regnet_y_400mf and densenet121:

  ```python
  model = torchvision.models.get_model(backbone, weights=weights)
  name, last = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
  model.set_submodule(name, nn.Linear(last.in_features, num_classes))
  ```

- **squeezenet:** its head is a `Conv2d`, not a `Linear`, so this fix doesn't cover it; remove it from the docstring.
- **Test:** parametrize over 3–4 backbones with `weights=None` and check the output shape.

### 1.8 The WebDataset config crashes at startup (high, ✅, S)

- **Where:** [base_trainer.py:199-206](src/trainers/base_trainer.py#L199-L206) and [webdataset_dataloader.py:17-28](src/dataloaders/webdataset_dataloader.py#L17-L28)
- **Problem:** a `generator` kwarg is always injected, but `WebDatasetDataLoader` doesn't accept it, so it raises `TypeError`. The documented `classifier_webdataset_cpu_config.yaml` cannot run.
- **Fix:** the proper fix is in 5.5. If WebDataset must work before then, accept and pass through `generator`.

### 1.9 CLI overrides: `1e-3` is parsed as a string, and `--mode` overrides the YAML (medium, ✅, S, R+)

- **Where:** [config_parser.py:16-27](src/config_parser.py#L16-L27), [config_parser.py:131-138](src/config_parser.py#L131-L138), [train.py:39-42](train.py#L39-L42) (and the same flag in test, inference and export)
- **Problem:**
  - PyYAML parses `1e-3` as a string, so `-o optimizer.args.lr=1e-3` makes SGD raise TypeError.
  - The `--mode` default silently overrides both the YAML value and `-o mode=…`.
- **Fix:**
  - Replace `parse_omegaconf_primitive` and the loop with `config.merge_with_dotlist(args.override)` under struct mode. Verified on omegaconf 2.3:
    - `1e-3` becomes `0.001`.
    - `model.layers[2]=512` and whole-list overrides still work.
    - Unknown keys are still rejected.
  - Make the `--mode` default `None`, or remove `mode` entirely (7.2).
- **Test:** `-o optimizer.args.lr=1e-3` yields the float `0.001`.

### 1.10 Class indices are computed separately for each split (medium, ✅, S)

- **Where:** [base_dataset.py:19-36](src/datasets/base_dataset.py#L19-L36), called per split from [classifier_dataset.py:76-85](src/datasets/classifier_dataset.py#L76-L85)
- **Problem:** if `test/` lacks one class folder, train maps `{ant:0, bee:1, cat:2}` but test maps `{ant:0, cat:1}`, and evaluation is silently mislabeled. The same applies to `torchvision.datasets.ImageFolder`.
- **Fix:** `assert val_set.classes == train_set.classes`, and the same for test.

### 1.11 Data-prep scripts that lose data or aren't reproducible (medium, ✅, S)

- **Uppercase extensions:**
  - [train_val_test_split.py:65-66](scripts/train_val_test_split.py#L65-L66) and [duplicate_data.py:64-65](scripts/duplicate_data.py#L64-L65) silently drop `.JPG` (20 images in, 10 copied).
  - Inference rejects `.PNG` ([classifier_trainer.py:408-413](src/trainers/classifier_trainer.py#L408-L413)).
  - Fix: compare lower-cased extensions.
- **Non-reproducible splits:** [train_val_test_split.py:58-66](scripts/train_val_test_split.py#L58-L66) shuffles an *unsorted* `glob`, so the same seed gives different splits on different filesystems. Fix: `sorted(glob(...))`.
- **`-r False` still deletes:** [find_corrupt_images.py:43-49](scripts/find_corrupt_images.py#L43-L49) uses `type=bool`, so `-r False` still removes files after the prompt. Fix: `action="store_true"`.
- **Directory name mismatch:** the split script creates `val/`, but every config expects `valid/`. Pick one.

### 1.12 `num_workers: 0` crashes because of `prefetch_factor: 2` (medium, ✅, S)

- **Where:** [base_dataloader.py:32](src/dataloaders/base_dataloader.py#L32) and the cpu, gpu and mnist configs
- **Problem:** this combination raises ValueError, yet [base_trainer.py:195](src/trainers/base_trainer.py#L195) tells macOS users to set `num_workers: 0`.
- **Fix:** drop `prefetch_factor` from the configs and the wrapper and use the `DataLoader` defaults.

### 1.13 A missing `git` binary crashes every run (low; becomes high with the slim Docker image in 3.5) (✅, S)

- **Where:** [common.py:96-102](src/utils/common.py#L96-L102)
- **Problem:** only `CalledProcessError` is caught; a missing binary raises `FileNotFoundError`.
- **Fix:** catch `(subprocess.CalledProcessError, OSError)`.

---

## Phase 2: Delete dead and broken code

These are pure deletions with no behavior change on the main path. Their tests go with them. This is about 1,400 lines in total.

**Status (2026-10-27): done.**

- **Tests:** 93 passed, 25 skipped (CUDA-only), on both the exact lock and the latest stack. The 34 fewer passing tests are the tests of deleted code.
- **Smoke runs passed:**
  - MNIST: train → `test.py` → `inference.py` → both ONNX exports
  - the binary classifier with ROC/PR/calibration plots
  - the WebDataset config with TensorBoard logging on
- **MNIST is unchanged within noise with PyTorch's default init:** 96.6% val and 96.7% test after 2 epochs, vs 96.2% and 96.8% before.
- **Changes beyond the table:**
  - The loss registry keeps only `NLLLoss` and `CrossEntropyLoss`. BCE, BCEWithLogits and MSE can't work with log-probability outputs and class-index targets.
  - `top_k_accuracy_score` was fixed rather than deleted: `calc_metric` now passes it `y_score`, with one test.
  - Deleting base_metrics.py also removed `optimal_thres_from_roc_curve` and the tpr/tnr/ppv/npv/fpr/fnr/fdr metric names. `classification_report` covers the per-class rates.
  - The `quant_backend` and `tensorboard_port` config keys and the `-q` export flag were removed (6.2 is done).
  - The `.squeeze(-1)` on model outputs, which existed only for single-output BCE models, was removed.
- **Left for Phase 3:**
  - `torchinfo` is now unused.
  - `imageio` is used only by two data-prep scripts and the test fixtures.
  - `pyproject.toml` is untouched, because 3.2–3.3 rewrite it for uv.

| Delete | Where | ~Lines | Why |
| --- | --- | --- | --- |
| Feature importance and the `TRAIN_TEST_FEATSELECT` mode | [classifier_trainer.py:441-535](src/trainers/classifier_trainer.py#L441-L535), [train.py:41,65-67](train.py#L41) | 100 | Time-series leftover that crashes ✅ (`'list' object has no attribute 'shape'`) |
| `GradualWarmupScheduler` | [gradual_warmup_scheduler.py](src/schedulers/gradual_warmup_scheduler.py) | 206 | Never registered; crashes on torch ≥2.8 because `verbose` was removed ✅ |
| Bounding-box helpers | [image_operations.py](src/utils/image_operations.py) | 79 | Unused; the only `cv2` import in the repo |
| Visualization package | [utils/visualization/](src/utils/visualization/) | 190 | t-SNE, scatter and grid plots; unused |
| Custom weight init | [weights_initializer.py](src/utils/weights_initializer.py), [base_model.py:18-52](src/models/base_model.py#L18-L52) | 80 | Unneeded after 1.1; PyTorch's default init suits these models |
| Unused utils | [common.py](src/utils/common.py): `AverageMeter`, `MissingConfigError`, `try_bool`, `try_null`, `can_be_conv_to_float`, `capture_output`, `round_to_nearest_divisor`, `sigmoid`, `inherit_missing_dict_params`, `reorder_trainer_cfg`, `stable_sort`; `print_cuda_statistics`; the re-exports in [utils/\_\_init\_\_.py](src/utils/__init__.py) | 165 | Only referenced by their own tests |
| `NumerizeLabels` | [augmentations/numeric.py](src/augmentations/numeric.py) | 25 | Registered as a "transform" but never used |
| Hand-rolled metrics | [base_metrics.py](src/metrics/base_metrics.py) | 177 | `top_k_accuracy_score` and `accuracy_topk_torch` raise TypeError through `calc_metric` ✅. `comprehensive_clsf_metrics` returns a dict that breaks logging ✅. tpr/tnr/ppv/npv/fpr/fnr/fdr are sklearn's `recall_score`/`precision_score` with `pos_label` |
| MSSE loss and the BCE branches | [msse.py](src/losses/msse.py), BCE branches in [classifier_trainer.py:139-152,217-226](src/trainers/classifier_trainer.py#L139-L152) | 40 | `MSSELoss` returns a tuple the trainer can't use. Every model ends in LogSoftmax, so BCE can't work (sigmoid is applied to log-probs) ✅ |
| GAN-era dataset helpers | [classifier_dataset.py:118-151](src/datasets/classifier_dataset.py#L118-L151), [mnist_dataset.py:36-69](src/datasets/mnist_dataset.py#L36-L69) | 70 | `plot_samples_per_epoch`, `make_gif` and `finalize` are unused; deleting them drops `imageio` from the core deps |
| Placeholders | `data_mode: numpy` branches, `accimage_loader`, the time-series `c_out` warning ([classifier_trainer.py:41-48](src/trainers/classifier_trainer.py#L41-L48)) | 20 | Never implemented, or time-series leftovers |
| `__main__` demo blocks in library modules | [base_dataset.py:234-254](src/datasets/base_dataset.py#L234-L254), [classifier_model.py:84-92](src/models/classifier_model.py#L84-L92) (calls an undefined `Classifier`), [base_logger.py:65-71](src/loggers/base_logger.py#L65-L71), the scheduler files, tsne.py | 70 | Not tests, and partly broken; move anything useful into tests |
| TorchScript and quantization paths | [export_utils.py:82-110,156-197](src/utils/export_utils.py#L156-L197), [base_trainer.py:306-309,329-353](src/trainers/base_trainer.py#L329-L353), [export.py:44-47](export.py#L44-L47), `TEST_TORCHSCRIPT` ([classifier_trainer.py:112-114](src/trainers/classifier_trainer.py#L112-L114)) | 110 | Deprecated, and quantization is broken for both models; see 6.1–6.2 |
| In-process TensorBoard server and port scan | [base_trainer.py:277-288](src/trainers/base_trainer.py#L277-L288), `is_port_in_use` | 17 | Run `tensorboard --logdir` yourself, as the README already says |
| Log keyword filter | [base_logger.py:15-26](src/loggers/base_logger.py#L15-L26) (`LogFilter`, `console_exc_keyword`) | 18 | Unused feature |
| `BaseModel.print_summary` | [base_model.py:54-60](src/models/base_model.py#L54-L60) | 7 | Never called; drops the `torchinfo` dependency |
| Stale scripts | [scripts/inference.py](scripts/inference.py), [scripts/run_server.sh](scripts/run_server.sh) | 108 | Old API (AttributeError ✅), and calls other scripts with flags they don't accept |

---

## Phase 3: Dependencies, Python and tooling

**Status (2026-10-28): done.**

- **Tests:** 93 passed, 25 skipped (CUDA-only), on Python 3.11.18 and 3.14.10. Both environments were installed from `uv.lock` with `uv sync --locked --extra cpu`, as CI does.
- **Lint:** `ruff check` and `ruff format --check` pass with ruff 0.17.3, and so do all pre-commit hooks.
- **Docker:** both training images (CUDA and CPU) and the server image build.
  - The CPU image trains MNIST end to end: 96.5% val, 96.7% test.
  - The CUDA image ships torch 2.14.1+cu130 with CUDA 13.0 and cuDNN 9.24.
  - It wasn't run on a GPU: the build machine's GPUs were busy with other workloads.
- **Not verified here:** GitHub Actions itself, since nothing was pushed. Every workflow step was run locally.
- **Changes beyond the items below:**
  - `numpy` and `pillow` became direct dependencies because the code imports them.
  - The two data-prep scripts and the test tar helper use PIL instead of `imageio`.
  - uv 0.13.6 is pinned in CI and in both Dockerfiles; `uv.lock` was written with it. uv 0.11 can read it too.
  - The Docker images use Python 3.13 rather than 3.12. 3.14 runs DataLoader workers through forkserver, which made the test suite 1.7× slower; 5.4's persistent workers address that.

### 3.1 Python `>=3.11,<3.15`, with CI on 3.11 and 3.14 (S, R0)

- **Why move off 3.10:**
  - Python 3.10 reached end-of-life in October 2026.
  - The latest onnxruntime, scikit-learn, matplotlib, imageio and pillow all require ≥3.11.
- **Why stop below 3.15:** onnxruntime 1.31 and the PyPI torch 2.14 wheels have no Python 3.15 builds yet. (Corrected in Phase 3: the PyTorch index does have cp315 torch wheels.)
- **Evidence:** the suite passes on 3.13 and 3.14 with the latest dependencies ✅.
- **CI:** test only the bounds, 3.11 and 3.14. That is one fewer leg than today and covers the currently untested top end.
- **Also update:**
  - ruff `target-version`
  - the README badges

### 3.2 Bump direct dependencies; remove unused ones (M, R+)

- **Bump:** use the versions in [Appendix B](#appendix-b-dependencies). Key moves:
  - torch 2.8 → 2.14 and torchvision 0.23 → 0.29 (they move together)
  - onnxscript 0.4 → 0.7
  - webdataset 0.2 → 1.0
  - omegaconf 2.3 → 2.4
- **Remove:**
  - `six` and `packaging` (never imported)
  - `torchinfo` (Phase 2)
  - `opencv-python` and `aiofiles` (unused)
  - `imageio` (after Phase 2; the two scripts can use PIL)
  - `requests` (server-only; it goes away with 6.4)
- **Security floors:** keep the `urllib3`, `fonttools` and `filelock` floors only if they were added for advisories, and drop the `filelock <4` cap. (Done: they came with a February dependency update, so they are kept as uv `constraint-dependencies` rather than direct dependencies.)
- **Pin policy:**
  - Fast-moving ML libraries get `>=X.Y,<X.(Y+1)`.
  - Everything else gets `<next major`.
  - Exact versions live only in the lockfile.
- **New warnings to clean up after the bump:**
  - webdataset `shardshuffle`/`with_length` (5.5)
  - `dynamic_axes` (6.1)
  - TorchScript FutureWarnings (6.1)
  - pytest 9 `PytestRemovedIn10Warning` (tests/src/test_augmentations.py:74,78)

### 3.3 One source of truth for dependencies (M, R+; decided: uv)

- **Today there are four sources:**
  - `pyproject.toml`
  - `requirements/train.txt`, generated by a custom script with direct deps only
  - `tests/requirements.txt`
  - `server/requirements.txt`
- **This is already broken for pip users ✅.** `pip install -r requirements/train.txt` currently resolves onnxscript 0.4.0 with onnx-ir 1.1, which breaks ONNX export (3 tests fail). Installing from `poetry.lock` works.
- **Recommended: move to [uv](https://docs.astral.sh/uv/guides/integration/pytorch/).**
  - `[project.dependencies]` for training deps.
  - `[dependency-groups]` for `dev`, `test` and `server`.
  - CPU and CUDA torch selected with an extra: `uv sync --extra cpu` on laptops and CI, `uv sync --extra cu130` (or the CUDA build matching your driver) on GPU machines.
  - `uv.lock` replaces `poetry.lock`.
  - Delete the three requirements files and `scripts/poetry_to_pip_requirements.py`.
  - When someone needs a requirements file: `uv export --no-hashes > requirements.txt`.

  ```toml
  [project.optional-dependencies]
  cpu = ["torch>=2.14,<2.15", "torchvision>=0.29,<0.30"]
  cu130 = ["torch>=2.14,<2.15", "torchvision>=0.29,<0.30"]

  [tool.uv]
  conflicts = [[{ extra = "cpu" }, { extra = "cu130" }]]

  [tool.uv.sources]
  torch = [{ index = "pytorch-cpu", extra = "cpu" }, { index = "pytorch-cu130", extra = "cu130" }]
  torchvision = [{ index = "pytorch-cpu", extra = "cpu" }, { index = "pytorch-cu130", extra = "cu130" }]

  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true

  [[tool.uv.index]]
  name = "pytorch-cu130"
  url = "https://download.pytorch.org/whl/cu130"
  explicit = true
  ```

- **If you stay on Poetry:**
  - Generate `requirements/train.txt` from the lock including transitive pins (`poetry export --without-hashes`, from poetry-plugin-export).
  - Delete the custom export script.
  - In CI, install CPU torch from the PyTorch CPU index before everything else.

### 3.4 CI: CPU wheels, lint, two legs, offline tests (S, R+)

- **Where:** [.github/workflows/main.yml](.github/workflows/main.yml)
- **CPU wheels:** CUDA torch plus 14 `nvidia-*` packages plus triton is 3.91 GB per leg on a cache miss. The CPU wheel is 184 MB. The suite itself runs in about 27 s.
- **Matrix:** 3.11 and 3.14 (3.1).
- **Lint:** add `ruff check` and `ruff format --check` (3.6).
- **Unpinned or outdated actions:**
  - Drop `snok/install-poetry` (`version: latest` is unpinned) and `actions/cache@v3`.
  - With uv, use `astral-sh/setup-uv` with its built-in cache.
  - Bump `actions/checkout` and `actions/setup-python` to their current majors.
- **Downloads:** stop downloading MobileNetV2 weights and MNIST in tests (7.9). (Still open; CI downloads them.)

### 3.5 Docker: one CUDA stack on a slim base (S–M, R+)

- **Where:** [Dockerfile](Dockerfile)
- **Problem:**
  - `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` is 7.42 GB compressed, and its torch 2.5.1 goes unused.
  - Poetry installs torch 2.8 plus about 3.9 GB of CUDA wheels into `.venv` on top, so the image carries two CUDA stacks (est. ~11 GB).
- **Fix:**
  - PyPI torch wheels already bundle the CUDA runtime and cuDNN; the driver comes from the host through the NVIDIA Container Toolkit.
  - Use `FROM python:3.12-slim` with `uv sync --locked --extra cu130 --no-dev` (est. ~4 GB). (Done with `python:3.13-slim`. Measured: 3.23 GB compressed, about 6.2 GB unpacked, vs about 11.3 GB compressed before. A `TORCH_EXTRA=cpu` build arg gives a 0.41 GB CPU image. git was dropped, since `.dockerignore` excludes `.git`.)
  - Drop:
    - the apt dev tools (cmake, g++, vim, htop, bmon, iotop, tree, wget, bzip2, apt-utils)
    - the X11 libs for OpenCV, which isn't a training dependency
    - the unused `ARG CUDA`/`CUDNN`
    - `-G sudo` (sudo isn't installed)
    - the placeholder `LABEL maintainer`
  - Use a BuildKit cache mount instead of `poetry cache clear`.
  - Keep `git` only if you want commit hashes recorded (1.13 makes it optional).
  - [run_docker.sh](scripts/run_docker.sh) hard-codes `--gpus "device=0"`; make it optional.

### 3.6 A lint config that is actually enforced (S, R+)

- **Problem:** with the repo's own config, ruff 0.12.11 reports 1,447 violations ✅, and 77 of 80 files would be reformatted ✅.
  - About 1,200 of the violations come from the `D` (docstring style) and `ANN` (annotation) rules.
  - `ANN101`/`ANN102` no longer exist.
- **Fix:**
  - Keep `E, F, B, UP, I, SIM, RUF` (plus `N` if you like). Drop `D`, `ANN` and `T20`; logging cleanup (4.8) removes the prints from `src/` anyway. (Done, also keeping `C4` and `A` from the old config at the cost of one fix. `N` was dropped: most of its hits are PyTorch conventions such as `import torch.nn.functional as F`. `fix = true` was removed so `ruff check` fails in CI instead of rewriting files.)
  - Run `ruff format` once in its own commit, and list that commit in `.git-blame-ignore-revs`.
  - Bump `.pre-commit-config.yaml`. `pre-commit-hooks` is at v2.3.0 from 2019, and the ruff hook should match the dev-dependency version.
  - Run ruff in CI (3.4).

---

## Phase 4: Training procedure (modern PyTorch, kept minimal)

**Status (2026-10-28): done.**

- **Tests:** 87 passed, 27 skipped (CUDA-only), on Python 3.11 and 3.14.
  - New tests cover resume, AMP scores, warmup plus cosine, the eval transform per weights, train augmentation, seeding, duplicate log lines and the CUDA-to-CPU fallback.
  - The AMP and resume tests fail without their fixes ✅.
  - The count is below Phase 3's 93 because the tests of deleted code went with it.
- **MNIST smoke runs on CPU:**
  - Two seeded runs gave identical loss and accuracy every epoch.
  - Resume: a 2-epoch run, resumed from its `last.pth`, trained epochs 3 and 4 with the same LRs as an uninterrupted 4-epoch run (0.005 and 0.001464), and its TensorBoard step continued from 3376 ✅.
  - `test.py` on `best.pth` scored 97.6%, `inference.py` labelled 4 RGB PNGs correctly, and both ONNX exports match PyTorch ✅.
- **Not verified here:** bf16 autocast, `cudnn.benchmark` and deterministic algorithms on a GPU. The machine's shared GPUs were busy.
- **Changes beyond the items below:**
  - `dataset.preprocess` is one `type` plus `args`, instead of four `*_transform` keys that always named the same class.
  - `best.pth` has the same full-state format as `last.pth`. Test, inference and export load its `"model"` entry.
  - The WebDataset CPU config sets `use_amp: false`, since AMP now applies on CPU too.
  - Until 5.1, a val split carved from the train set (`validation_split > 0`) gets the train augmentation. The shipped classifier configs use a separate val dir, and MNIST has no augmentation. (Fixed in Phase 5.)
  - The server still passes `gpu_device`; it is rebuilt in Phase 6.
  - Five nitpicks in touched files are done; see [Nitpicks](#nitpicks-segregated).

### 4.1 Models return logits and use `CrossEntropyLoss` (S, R+)

- **Where:** every model ends in `LogSoftmax` with `NLLLoss` ([classifier_model.py:74](src/models/classifier_model.py#L74), [custom_model.py:49](src/models/custom_model.py#L49), [mnist_model.py:29](src/models/mnist_model.py#L29)).
- **Fix:**
  - Models return raw logits, trained with `nn.CrossEntropyLoss(label_smoothing=0.1)`. Label smoothing is optional; the torchvision classification recipe uses it. (Done. The three classifier configs set it; MNIST trains without it. `NLLLoss` left the loss registry, since it is wrong on logits.)
  - Softmax is applied only in metrics and inference.
  - Exported ONNX then outputs logits, which is what serving code expects.
- **Note:** this, together with 1.7, renames the classifier head's state-dict keys. Decided: old checkpoints don't need to load, so no key map.

### 4.2 Full-state checkpoints and real resume (S, R−, high value)

- **Where:** [base_trainer.py:104-112](src/trainers/base_trainer.py#L104-L112) and [classifier_trainer.py:105-119](src/trainers/classifier_trainer.py#L105-L119)
- **Problem:** only the weights are saved. `--resume_checkpoint` restarts at epoch 1 with fresh optimizer momentum and a fresh LR schedule.
- **Fix:**
  - Every epoch, save `last.pth = {"model", "optimizer", "scheduler", "epoch", "best_val_acc"}`, adding `"scaler"` when using fp16.
  - Save `best.pth` on improvement (1.2).
  - Resume restores all of it.
  - Load with `torch.load(path, map_location="cpu", weights_only=True)` and `model.load_state_dict(ckpt.get("model", ckpt))` so old bare state dicts still load. The full dict round-trips with `weights_only=True` for SGD with ReduceLROnPlateau and LambdaLR ✅. (Done without the bare-state-dict fallback, since old checkpoints don't need to load. Both files are saved after the scheduler step, so a resumed run continues with the next epoch's LR. `best_val_acc` is stored as a Python float, because `weights_only` rejects numpy scalars. Resume keeps the checkpoint's schedule, so a larger `trainer.epochs` doesn't stretch the cosine; past its end the LR rises again.)
  - Drop `weight_save_freq`; `last.pth` and `best.pth` cover it. (Done in Phase 1.)
  - Reference: PyTorch recipe "Saving and loading a general checkpoint".

### 4.3 Mixed precision done right (🔎, S, R0)

- **Where:** [classifier_trainer.py:144](src/trainers/classifier_trainer.py#L144)
- **Problem:**
  - It uses fp16 `autocast("cuda")` without a `GradScaler`, so gradients risk underflowing.
  - It is applied in training only.
  - It is a silent no-op on CPU configs that set `use_amp: true`.
- **Fix:**
  - Wrap forward and loss in both train and eval with `with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=cfg.use_amp):`. bf16 needs no scaler (Ampere+ GPUs and CPUs). (Done. Eval casts the outputs to float32 before the softmax, as numpy has no bfloat16.)
  - Only if pre-Ampere GPUs matter: use `dtype=torch.float16` with `torch.amp.GradScaler`.

### 4.4 Single-device training: drop DataParallel (S, R+, about −40 lines)

- **Where:** [classifier_trainer.py:79-83](src/trainers/classifier_trainer.py#L79-L83), [base_trainer.py:46-64](src/trainers/base_trainer.py#L46-L64), [base_trainer.py:88-98](src/trainers/base_trainer.py#L88-L98)
- **Why:** the PyTorch docs recommend DistributedDataParallel over DataParallel, even on a single node.
- **Fix:**
  - Delete DataParallel, the `module.` key renaming and the GPU-count checks. (Done in Phase 1, as part of 1.5.)
  - The config becomes a single `device: cuda` (or `cuda:1`, or `cpu`) instead of `device` plus a `gpu_device` list. (Done. Without CUDA, `cuda` falls back to CPU with a warning, as before. `--gpu_device` left the four scripts.)
  - Add DDP with `torchrun` (about 25 lines) only if multi-GPU becomes a requirement.

### 4.5 LR schedules from built-ins, with a default that does something (S, R+)

- **Problems:**
  - The shipped configs use ReduceLROnPlateau with patience 8, stepped only on validation epochs: `valid_freq` 2 over 12 epochs is 6 steps. It can never fire, so the LR is constant ✅.
  - `WarmupCosineAnnealingScheduler` ([warmup_cosine_annealing_scheduler.py:8-15](src/schedulers/warmup_cosine_annealing_scheduler.py#L8-L15)) trains epoch 1 at LR 0 and climbs again after `total_epochs` ✅.
- **Fix:**
  - Default to `CosineAnnealingLR`, with an optional linear warmup:

    ```python
    sched = getattr(torch.optim.lr_scheduler, cfg.type)(optimizer, **cfg.args)
    if cfg.warmup_epochs:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs)
        sched = SequentialLR(optimizer, [warmup, sched], milestones=[cfg.warmup_epochs])
    ```

  - This matches the custom scheduler from epoch 2 on, without the LR=0 epoch ✅.
  - Keep the ReduceLROnPlateau `step(metric)` special case.
  - Drop the OneCycleLR per-batch special case unless you use it.
  - (Done. All four configs use `CosineAnnealingLR` with `T_max: ${trainer.epochs}` and `warmup_epochs: 0`, and any `torch.optim.lr_scheduler` class works from the config. `WarmupCosineAnnealingScheduler` and the OneCycleLR case are deleted.)

### 4.6 Real training augmentation; one eval transform everywhere (S, R+)

- **Where:** [classifier_transforms.py:7-19](src/augmentations/classifier_transforms.py#L7-L19)
- **Problem:**
  - Train, val and test use the same Resize 256 → CenterCrop 224, so there is no augmentation.
  - The ImageNet constants are copied by hand.
- **Fix:**
  - **Train:** `RandomResizedCrop(224) + RandomHorizontalFlip()`. It is also about twice as cheap as the full-image resize: 1,351 vs 665 img/s per core ✅.
  - **Eval** (val, test, inference, export, server), for pretrained backbones: `weights.transforms()` from the torchvision weights enum. Resize, crop and normalization then match the checkpoint; for example, V2 weights resize to 232. (Done. The configs pass `weights: ${model.args.pretrained_weights}`, and the classes expose `train_transform` and `eval_transform`. For MobileNetV2 the eval output is identical to the old transform ✅.)
  - **MNIST:** keep its constants and add `Grayscale()`.
  - **API:**
    - Stay on the `torchvision.transforms` v1 API. It shows no deprecation warnings on 0.29 ✅, and v2 tensor ops measured slower on CPU.
    - Switch to `transforms.v2` only if you want MixUp/CutMix.
    - Import from `torchvision.transforms`, not the private `torchvision.transforms.transforms`.

### 4.7 Seeding and reproducibility (S, R+)

- **Where:** [config_parser.py:58-73](src/config_parser.py#L58-L73)
- **Problem:**
  - Seeding uses ad-hoc +1/+2/+3 offsets and runs inside the config constructor.
  - `reproducible: true` in every config disables `cudnn.benchmark`, which is slower 🔎, yet runs still aren't fully deterministic: `cudnn.deterministic` only covers convolutions.
- **Fix:**
  - Add a 5-line `seed_everything(seed)` covering random, numpy and torch, called from `train.py`. `torch.manual_seed` already seeds every CUDA device.
  - Default to `reproducible: false`, with `cudnn.benchmark = not reproducible`.
  - Reproducible mode adds `torch.use_deterministic_algorithms(True, warn_only=True)`. (All three done, in `train.py`.)
  - The seeded DataLoader generator is already correct ✅.

### 4.8 Logging and TensorBoard (S, R+)

- **Problems:**
  - `get_logger` ([base_logger.py:29-62](src/loggers/base_logger.py#L29-L62)) adds new handlers on every call, which duplicates every log line ✅.
  - `src/` contains 26 `print`s.
  - ANSI color codes (`BColors`) end up in the log files.
- **Fix:**
  - Configure logging once in each entry script: `logging.basicConfig(handlers=[StreamHandler(), FileHandler(run_dir / "train.log")], ...)`.
  - Modules use `logging.getLogger(__name__)`.
  - No prints or colors in `src/`.
  - (Done with a short `setup_logging(log_path)` that each entry script calls; runs log to `logs/train.log`, `test.log`, `inference.log` or `export.log`. The trainers keep `logging.getLogger(logger_name)` until 7.4. `export_utils` still prints when it gets no logger; Phase 6 rewrites it.)
- **TensorBoard:**
  - Log the config once as YAML text. Today it is one `add_text` per key ([base_trainer.py:273-275](src/trainers/base_trainer.py#L273-L275)).
  - Write per-iteration scalars every `batch_log_freq` steps.
  - Log images from the last training batch (5.3).
  - (All three done. The per-iteration step is computed from the epoch, so it continues after a resume.)

---

## Phase 5: Data pipeline and speed

**Status (2026-10-28): done.**

- **Tests:** 71 passed, 27 skipped (CUDA-only), on Python 3.11 and 3.14.
  - New tests cover the val split's eval transform, the MNIST split sizes, shards that keep the image files and mix the classes, and DataLoader workers that read every sample once, including a worker without a shard.
  - The val-split and WebDataset tests fail without their fixes ✅.
  - The count is below Phase 4's 87 because the 19 tests of deleted code went with it: the custom loaders, the dataset helpers and MNIST's `data_mode`.
  - The suite takes about 29 s on 3.11 and 41 s on 3.14, vs 56 s and 88 s after Phase 4.
- **`src/`** is 1,392 lines, down from 1,770 after Phase 4.
- **Measured on CPU ✅:**
  - MNIST, 3 epochs: average epoch time 4.07 → 3.88 s on Python 3.11, and 5.28 → 4.78 s on 3.14.
  - `--verbose` label counts: the same counts, in 0.11 s vs 1.41 s for a pass over MNIST.
  - Startup: `import src.trainers` takes 2.6 s, down from 3.0 s.
  - WebDataset: see 5.5.
- **Smoke runs on CPU:**
  - MNIST: a run resumed from `last.pth` trained epochs 3 and 4 with the same LRs as an uninterrupted run (0.005 and 0.001464). `test.py` scored 97.5% on `best.pth`, `inference.py` labelled 4 RGB PNGs correctly, and both ONNX exports match PyTorch ✅.
  - WebDataset: `train.py` trained, validated and tested from shards with 4 workers, on Python 3.11 and 3.14 ✅.
- **Not verified here:** the GPU effect of 5.2 and of pinned memory. The machine's shared GPUs were busy.
- **Changes beyond the items below:**
  - Config: `dataloader` holds only `batch_size` and `num_workers`, and `validation_split` moved to `dataset.args`. The trainer sets shuffling, `drop_last`, pinned memory and persistent workers itself, so `type`, `shuffle`, `drop_last`, `pin_memory` and `worker_init_fn` are gone.
  - The dataset classes no longer take `**kwargs`, so unknown `dataset.args` keys raise. The MNIST config drops `train_path`, `val_path` and `test_path`, which it never read, and `MnistDataset` drops its one-value `data_mode`.
  - `--verbose` no longer changes training. Its label count iterated the train loader, which shifted the shuffle: epoch-1 loss 0.698 vs 0.776 on MNIST ✅.
  - The train loss is summed in float32 on the device, so the logged value can differ in the 4th decimal (0.7706 vs 0.7705 on MNIST).
  - The trainer tests' fixture creates its own split dirs. Before, those tests passed only after the dataset tests had created them.

### 5.1 Built-in datasets and loaders (M, R+, about −320 lines)

- **Built-in replacements:**
  - [base_dataset.py](src/datasets/base_dataset.py) (254 lines) re-implements `torchvision.datasets.ImageFolder`, which supports the same layout (nested class sub-folders included) and has `.classes` and `.targets`.
  - `CustomDataLoader` ([base_dataloader.py](src/dataloaders/base_dataloader.py), 90 lines) exists only to carve a validation split. Use `torch.utils.data.random_split(..., generator=g)` or `Subset` with a plain `DataLoader`.
  - A split carved from the train set inherits the train augmentation. For MNIST, build a second `MNIST(train=True, transform=eval_tf)` and take `Subset`s of both with the same indices.
- **Stop swallowing exceptions:**
  - `__getitem__` catches every exception, prints it and returns `None` ([base_dataset.py:158-171](src/datasets/base_dataset.py#L158-L171)). The collate function then drops it ([base_dataloader.py:9-14](src/dataloaders/base_dataloader.py#L9-L14)).
  - Batches silently shrink and the bad path is never logged.
  - Let it raise, and clean the data once with `scripts/find_corrupt_images.py`.
- **Also:** add the class-list assert from 1.10.
- (Done. torchvision's `ImageFolder` and plain `DataLoader`s replace them; `src/dataloaders/` and `base_dataset.py` are deleted. The val split uses `random_split`, then points the val `Subset` at a second dataset with the eval transform. Corrupt images now raise, and the README runs `find_corrupt_images.py` first. The class check from Phase 1 also covers WebDataset splits now.)

### 5.2 Remove per-step host↔device syncs (🔎, S, R0)

- **Where:** [classifier_trainer.py:147,153,168-171](src/trainers/classifier_trainer.py#L147-L171)
- **Problem:** three `.item()` calls per step block the CPU until the GPU drains. Est. 5–20% of step time for MobileNetV2-size models at batch size 32.
- **Fix:**
  - Accumulate `loss.detach() * bs` and `(pred == target).sum()` as tensors, with a single `.item()` at epoch end.
  - Write per-iteration scalars every `batch_log_freq` steps. (Done in Phase 4, 4.8.)
  - Do this together with 1.4.
- (Done. Loss and correct counts stay on the device until the epoch ends; the per-iteration loss still calls `.item()` every `batch_log_freq` steps.)

### 5.3 Stop re-reading the data (S, R+)

- **Mean/std auto-computation** (deleted in 1.6):
  - It makes two full decode passes over the train set on every train, test and export run.
  - That costs 3.6 s per 1,000 images ✅, about 77 minutes per invocation for ImageNet-1k (extrapolated).
  - `export.py` took 20.9 s vs 11.8 s on 1,000 images ✅.
- **`--verbose` label counting** ([base_trainer.py:229-256](src/trainers/base_trainer.py#L229-L256)):
  - It iterates all three DataLoaders, a full decode pass, just to count labels.
  - That takes 0.72 s, vs 0.24 ms reading `dataset.targets` ✅.
- **TensorBoard image logging** ([classifier_trainer.py:192-195](src/trainers/classifier_trainer.py#L192-L195)):
  - `add_images(next(iter(train_loader)))` respawns the workers and decodes `num_workers × prefetch_factor` batches every epoch, to keep one.
  - That costs 0.25 s per epoch ✅, about 35–40% of a small epoch's loading time.
  - Log the last batch from the loop instead. (Done in Phase 4, 4.8.)
  - It also logs normalized tensors, which look clipped. (Done: `make_grid(normalize=True)` rescales them. Before, 72.5% of the pixel values were clipped ✅.)
- (Label counts done: they come from `targets`, indexed by a `Subset`'s indices for a carved val split.)

### 5.4 DataLoader settings (S, R0)

- **`persistent_workers: true`** is set nowhere today, so workers respawn for every loader, every epoch:
  - 0.25 s per loader start with fork on Linux ✅
  - about 3.2 s with spawn on macOS and Windows ✅
  - On Python 3.14 the default start method is forkserver; the suite took 102 s vs 27 s mostly because of this ✅.
- **`pin_memory`:** only when training on CUDA.
- **`prefetch_factor`:** drop it (1.12).
- **`drop_last: true` for the train loader:** avoids BatchNorm failing on a final batch of 1.
- (All done; the trainer sets them, not the config. For a WebDataset, `drop_last` applies per worker, so each drops its own partial batch, different images each epoch.)

### 5.5 WebDataset done right (M, R+; decided: keep it)

- **(a) Re-encoding:** the converter ([convert_dataset_to_tar.py:37-52](scripts/convert_dataset_to_tar.py#L37-L52)) writes one class-sorted tar and re-encodes the JPEGs at quality 100 ✅:
  - the tar is 2.2× larger (132 vs 61 MB)
  - decode is 1.8× slower (812 vs 1,497 img/s per core)
  - and quality is lost
  - Fix: write shuffled samples with `wds.ShardWriter("…-%06d.tar", maxcount=…)` that store the original file bytes.
- **(b) Single shard:** with one shard, only one DataLoader worker gets data, which is why the config pins `num_workers: 0`. Throughput was 341–350 img/s vs 1,220–1,320 img/s with 4 shards × 4 workers ✅.
- **(c) Class-sorted batches:** class-sorted shards with `shuffle(1000)` mean the first batches of each epoch contain 1–3 of 20 classes ✅.
- **(d) Length counting:** `_get_webdataset_len` ([classifier_dataset.py:15-22](src/datasets/classifier_dataset.py#L15-L22)) decodes every image on every run. That is 1.24 s per 1,000 samples, vs 0.054 s scanning tar headers ✅. Write the sample count at conversion time instead.
- **(e) Loader details:**
  - `len(loader)` returns samples, not batches (20 vs 5 ✅), so progress % and tqdm totals are wrong.
  - Shuffle only the train split.
  - Set `shardshuffle` explicitly; `True` warns on webdataset 1.0.
- **Loader class:** replace `WebDatasetDataLoader` with `wds.WebLoader`, or `DataLoader(ds.batched(bs), batch_size=None)`. 1.8 was fixed in Phase 1 by passing `generator` through; this replaces that stopgap.
- (Done, measured on 1,000 synthetic train JPEGs ✅:
  - (a) The converter writes a dir of shuffled shards per split with the original file bytes: 58 MB vs 126 MB.
  - (b) 4 shards with 4 workers load 1,557 img/s vs 373–430 img/s. `empty_check=False` lets a worker without a shard read nothing, e.g. on a one-shard val split.
  - (c) The first batch holds 17 of 20 classes, vs 3.
  - (d) `info.json` holds the image count and class names, so setup takes 0.00 s vs 1.2 s.
  - (e) A plain `DataLoader` batches, so `len()` counts batches; only the train split shuffles.
  - Also: wds' `with_length()` creates a class at runtime that pickle can't find, so DataLoader workers failed under spawn (macOS) and forkserver (Python 3.14) ✅. A 3-line subclass defines `__len__` instead; the config had pinned `num_workers: 0`.
  - Old single-tar datasets need converting again. The converter's `--mp` mapping file is gone, as `info.json` has the class names.)

### 5.6 Faster startup imports (S, R+)

- **Problem:** `import src.trainers` takes 3.2 s, vs 1.6 s for torch and torchvision alone. [utils/\_\_init\_\_.py](src/utils/__init__.py) eagerly imports export utils (onnx, onnxruntime), visualization (sklearn.manifold) and pyplot ([metrics/plots.py:1-4](src/metrics/plots.py#L1-L4)).
- **Fix:** empty that `__init__` and import pyplot inside the plot functions. Measured: 3.2 → 2.5 s ✅.
- (Done. The `__init__` was already empty after Phase 2; moving pyplot into the plot functions took `src.metrics` from 0.87 to 0.57 s, and `import src.trainers` from 3.0 to 2.6 s ✅.)

### 5.7 Optional: batched inference (R−, about +8 lines)

- **Where:** `inference()` runs one image at a time with a sync per image ([classifier_trainer.py:421-432](src/trainers/classifier_trainer.py#L421-L432)).
- **Impact:** no gain on CPU (30 vs 30 img/s ✅); est. 5–8× on GPU at batch size 32 🔎.
- **Fix:** only if bulk GPU inference matters: a Dataset over the file paths, a `DataLoader`, and `torch.inference_mode()`.
- (Skipped: 6.3 rewrites inference.)

---

## Phase 6: Export, inference and serving

### 6.1 One export path: ONNX through the dynamo exporter (S, R+, about −120 lines)

- **Where:** [export_utils.py:13-175](src/utils/export_utils.py#L13-L175), [base_trainer.py:291-327](src/trainers/base_trainer.py#L291-L327)
- **Problem:** there are four modes (`ONNX_TS`, `ONNX_DYNAMO`, `TS_TRACE`, `TS_SCRIPT`) behind a Strategy ABC with four classes. On torch 2.14:
  - TorchScript `trace`, `script`, `save` and `load` raise FutureWarnings pointing to torch.export ✅.
  - `torch.onnx.export` has defaulted to `dynamo=True` since 2.9, so `ONNX_TS` silently runs the dynamo exporter anyway ✅.
  - `dynamic_axes` is deprecated in favor of `dynamic_shapes` ✅.
- **Fix:** replace it all with one function. Verified on torch 2.14.1: a batch of 5 runs in onnxruntime and matches PyTorch ✅.

  ```python
  def export_onnx(model, sample, path):
      program = torch.onnx.export(model.eval(), (sample,), dynamo=True, dynamic_shapes=({0: "batch"},))
      program.save(path)
  ```

- **Parity check:** keep one `np.testing.assert_allclose` check. Drop the 32-run benchmark, which takes 30–55% of export time ✅.
- **Optional PyTorch-native artifact:** `torch.export.save(torch.export.export(model, (sample,), ...), "model.pt2")` ✅.
- **Also:**
  - Delete the `TEST_TORCHSCRIPT` mode of `test.py`. (Done in Phase 2, with `TS_TRACE` and `TS_SCRIPT`; `ONNX_TS` and `ONNX_DYNAMO` remain.)
  - Requires onnxscript ≥0.7 (3.2).

### 6.2 Quantization: delete the `torch.ao` path (S, R+, about −60 lines; deletion done in Phase 2)

- **Problem:**
  - Eager `torch.ao.quantization` with a QuantStub wrapper and no layer fusion fails for both shipped models ✅:
    - MnistModel fails with a `view` error.
    - MobileNetV2 fails with `aten::add.out` on QuantizedCPU.
  - `torch.ao.quantization` is deprecated in favor of torchao, and warns at runtime ✅.
- **Fix:**
  - Delete `-q`, `quant_backend`, `QuantizedModelWrapper` and `quantize_model`.
  - If INT8 CPU serving is ever needed, run `onnxruntime.quantization.quantize_static` on the exported ONNX. onnxruntime is already a dependency and it's about 15 lines.
  - That gave 2.1× throughput on the default model ✅; accuracy has not yet been checked on real data.

### 6.3 Export and inference without the Trainer (S–M, R+)

- **Problem:**
  - `export.py` builds the whole training stack just to export: datasets read from disk, the mean/std pass, dataloaders, optimizer and scheduler.
  - `inference.py` and the server go through `ClassifierTrainer`, which returns early when `mode == "INFERENCE"` ([base_trainer.py:145-146](src/trainers/base_trainer.py#L145-L146), [classifier_trainer.py:54-59](src/trainers/classifier_trainer.py#L54-L59)).
- **Fix:**
  - Export and inference need only `model = build_model(cfg)`, `model.load_state_dict(...)` and the eval transform.
  - Save the class names (`train_set.classes`) in the run's `config.yaml` so inference can print names instead of indices.
  - Inference should:
    - lower-case extensions (1.11)
    - use `torch.inference_mode()`
    - report softmax probabilities

### 6.4 Server: rebuild a minimal one on ONNX Runtime (M, R+; decided: keep the server)

- **Problem:**
  - **It can't start ✅.** `CustomDictConfig(resume=...)` raises TypeError at import ([inference_server.py:42](server/inference_server.py#L42)); fixing that hits a missing `resume_checkpoint` key.
  - **Its image installs only `server/requirements.txt`**, so torch and omegaconf are missing ✅.
  - **Endpoints block the event loop.** They are `async def` but do blocking file I/O, inference, `urlopen` without a timeout and `requests.post`, so requests serialize and one slow URL hangs the server.
  - **Confusing routing:** the enum names `ResNet50`/`ResNet101` both point to the same MNIST checkpoint, and the `{inputModel}` path parameter is unused.
- **Fix:** rewrite it rather than patch it, as a FastAPI app of about 40 lines:
  - Load `model.onnx` with onnxruntime at startup.
  - Expose one sync `def predict(file: UploadFile)`; FastAPI runs sync endpoints in a threadpool.
  - Return the top-k class names and probabilities.
  - Bake normalization into the exported graph, or store input size, mean and std as ONNX metadata, so serving can't drift from training.
  - Drop the callback/`back_url`/URL-download machinery.
- **Result:**
  - ONNX Runtime fp32 ran 2.7× faster than eager PyTorch on CPU ✅.
  - The server no longer imports the training stack.

### 6.5 Server image and scripts (S, R+)

- **Image:** `python:3.12-slim` with fastapi, uvicorn, python-multipart, onnxruntime, pillow and numpy. Est. ~0.3 GB, vs 7.4 GB for the current devel base.
- **Scripts:**
  - [run_server_docker.sh](server/run_server_docker.sh) starts `bash` instead of the server; fix it.
  - Fix the README usage for both server scripts (7.10).

---

## Phase 7: Structure and readability

### 7.1 Config: plain OmegaConf, with side effects in the entry scripts (S, R+, about −100 lines)

- **Problem:**
  - `CustomDictConfig` ([config_parser.py:30-143](src/config_parser.py#L30-L143)) subclasses `DictConfig`. Its constructor:
    - seeds the RNGs (moved to `train.py` in Phase 4, 4.7)
    - flips the cuDNN flags (moved to `train.py` in Phase 4, 4.7)
    - shells out to git
    - creates directories
    - writes `config.yaml`
  - `BaseTrainer` then writes `config.yaml` again after mutating `dataset.args` ([base_trainer.py:175-186](src/trainers/base_trainer.py#L175-L186)).
  - `from_args` copies every CLI argument into the config as a top-level key, so `resume_checkpoint` exists both at top level and under `trainer`.
- **Fix:**
  - Add `load_config(path, overrides) -> DictConfig`, about 8 lines: `OmegaConf.load`, `set_struct`, then `merge_with_dotlist`.
  - In `train.py`, explicitly:
    - call `seed_everything(cfg.seed)`
    - create the run directory: `run_dir = make_run_dir(cfg)`
    - save the config once: `OmegaConf.save(cfg, run_dir / "config.yaml")`

### 7.2 Entry scripts: share arguments and drop the `mode` plumbing (S, R+)

- **Problem:**
  - train, test, inference and export repeat about 45 identical argparse lines.
  - `--dev`, `--gpu_device` and `--mode` duplicate what `-o key=value` already does.
  - `mode` strings switch control flow inside constructors (INFERENCE, TEST_TORCHSCRIPT, TRAIN_TEST_FEATSELECT).
- **Fix:**
  - One small `add_common_args(parser)` with `--cfg`, `-o`, `--resume`/`--weights` and `-v`.
  - Each script's `main()` calls exactly what it needs: `trainer.fit()` then `evaluate(best)`, or `export_onnx(...)`, or `predict(...)`.
  - Optional: rename `test.py` to `evaluate.py` so it reads less like a pytest file.

### 7.3 Registries become built-ins plus plain dicts (S, R+)

- **Problem:** there are 8 copies of "a dict plus an `init_*` function with try/except", one each for trainers, transforms, dataloaders, datasets, models, losses, optimizers and schedulers. Each hand-lists a subset of what torch already provides. (The dataloader one went with the custom loaders in Phase 5.)
- **Fix:**
  - Built-ins straight from the config: `getattr(torch.optim, cfg.optimizer.type)(model.parameters(), **cfg.optimizer.args)`, and the same for `torch.nn` losses and `torch.optim.lr_scheduler`.
  - Every built-in then works from YAML, and `src/optimizers/`, `src/losses/` and `src/schedulers/` disappear.
  - Project code (models, datasets) gets one plain dict each, indexed directly. A `KeyError` names the missing key.

### 7.4 One `Trainer` class (L, R+, about −500 lines est.)

- **Problem:**
  - `_BaseTrainer` → `BaseTrainer` → `ClassifierTrainer` is three classes (about 890 lines) for one concrete trainer, with `mode`-dependent early returns in two constructors.
  - `BaseTrainer.__init__` alone is about 150 lines of dataset, dataloader and TensorBoard setup.
- **Fix:**
  - One `Trainer` (est. 250–300 lines) that receives the model, loaders, optimizer, scheduler and loss.
  - It owns `fit()`, `validate()`, `evaluate()`, `save()` and `load()`.
  - Data building moves to `build_dataloaders(cfg)`.
  - A new task type later means a new small trainer, not a deeper hierarchy.

### 7.5 Transforms as functions, not classes (S, R+)

- **Problem:** the preprocess "classes" are namespaces: their transforms are class attributes built at import time (shared mutable state, ruff RUF012), with an empty `__init__`. They are instantiated only to read those attributes. (Phase 4 builds them in `__init__` as `train_transform` and `eval_transform`, since the eval transform now depends on the weights.)
- **Fix:** `build_transforms(cfg) -> (train_tf, eval_tf)`.

### 7.6 Models (S, R+)

- **Base class:** subclass `nn.Module` directly. `BaseModel` adds only trivial `__init__`/`forward` overrides and the init helpers deleted in 1.1.
- **`ClassifierModel`:**
  - Make it a function, `torchvision_classifier(backbone, num_classes, weights)`, using the helper from 1.7.
  - Either fix `feat_extract` or delete it.
- **`CustomModel`:** replace the hand-computed flattened size ([custom_model.py:39-40,55-75](src/models/custom_model.py#L55-L75)) with `nn.AdaptiveAvgPool2d(1)`, which also makes it input-size agnostic.
- **`MnistModel`:** use `torch.flatten(x, 1)` instead of `x.view(-1, 320)`.

### 7.7 Metrics (S, R+)

- **What's left:** after Phase 2, `src/metrics/` is a dict mapping names to sklearn functions, plus three plot functions.
- **Plots** ([plots.py](src/metrics/plots.py)) should:
  - take `y_score` (1.3)
  - draw on a passed-in `ax`, via `Display.from_predictions(..., ax=ax)`
  - close their figure; today each call leaks one figure ✅
- **`calc_metric`:** take `average` from the config instead of guessing it from `np.unique(y_true)`, where a val split missing one class flips binary to macro.

### 7.8 Config files (S, R+)

- **Unused or runtime-only keys to remove:**
  - `verbosity` (never read)
  - `git_hash` and `mode` (runtime state; `mode` done in Phase 1)
  - `quant_backend` (done in Phase 2)
  - `tensorboard_port` (done in Phase 2)
  - `weight_save_freq` and `save_best_only` (done in Phase 1)
  - `gpu_device` (done in Phase 4, 4.4)
  - `prefetch_factor` (done in Phase 1)
- **Duplicates:**
  - `dataset.num_classes` vs `model.args.num_classes`: use `${model.args.num_classes}` interpolation or delete one.
  - `model.info` input sizes vs the transforms.
- **Contradictions:**
  - The webdataset config sets `device: cpu` together with `use_amp: true`, `pin_memory: true` and `gpu_device: [0]`. (`use_amp` and `gpu_device` done in Phase 4, `pin_memory` in Phase 5.)
  - `configs/README.md` says `feat_extract` freezes the backbone; it actually swaps the head for Identity. (Done in Phase 1.)
  - It also says "mode: TRAIN or TEST".
- **Guard against drift:** add a tiny parametrized test that loads every `configs/*.yaml` and builds its model and transforms.

### 7.9 Tests (M, R+)

- **Regression tests:** add one per Phase 1 fix. The 119 passing tests catch none of them ✅. (Done in Phase 1.)
- **Temp files:** replace the hard-coded `/tmp/pytest` root ([tests/conftest.py:24](tests/conftest.py#L24)) with pytest's `tmp_path`. The shared root also hid an ordering dependency: the trainer tests used split dirs that only the dataset tests created (fixed in Phase 5).
- **Offline and fast:**
  - Problem: trainer tests download MobileNetV2 weights and train it at 224 px for 4 epochs, and MNIST tests download MNIST.
  - Fix: use `pretrained_weights: null`, a tiny model and `torchvision.datasets.FakeData`.
- **No ordering dependency:** `pytest-order` exists only because `test_inference` consumes `test_export`'s output file ([test_export_utils.py:65](tests/src/test_utils/test_export_utils.py#L65)). Merge them into one test and drop the dependency.
- **Global monkeypatch:** `test_get_git_revision_hash` patches `subprocess.check_output` globally and never restores it; use the `monkeypatch` fixture. (Done in Phase 1, with `mocker`.)
- **Cleanup:** delete the tests for deleted code.

### 7.10 README and docs: commands that work (S, R+)

- **Commands that fail as written ✅:**
  - `pip install -r requirements.txt`: the file is `requirements/train.txt` (moot after 3.3).
  - The split script's `-ts` flag should be `--ts`.
  - The split script doesn't write the class map the README promises.
  - `duplicate_data.py --rd/--td`: the script defines `-rd/-td`.
  - `--cfg custom_classifier_cpu_config.yaml` is missing the `configs/` prefix.
  - The server scripts' `-m` and `-h/--http` flags don't exist.
- **Other doc errors:**
  - The TensorBoard port is given as both 6007 and 6006.
  - It claims graph logging, but no graph is logged.
  - The "Inference" section is empty.
  - The Python badges are out of date.
- **Add** a 10-line "How a training run flows" section: `train.py` → `load_config` → build data and model → `Trainer.fit` → evaluate `best.pth`.

### 7.11 Optional: flatten the package layout (L, R+)

- **Problem:** there are 11 sub-packages, most holding one file plus a registry, for about 2,000 lines of code.
- **Target:** about 7 modules ([Appendix C](#appendix-c-target-layout-optional)).
- **When:** do it last, or never. Phases 1–6 deliver most of the value; this mostly makes navigation easier.

---

## Nitpicks (segregated)

These are low impact. Fold each into whichever PR touches the file. Many become moot once the code they're in is deleted.

- [classifier_trainer.py:1-2](src/trainers/classifier_trainer.py#L1-L2): the docstring says "Time Series Classification trainer". Line 41 mentions a "UCR Dataset". (Done in Phase 2.)
- [classifier_trainer.py:327](src/trainers/classifier_trainer.py#L327) logs `val_end_tm - val_end_tm`, so validation time always reads 0.000 s. (Done in Phase 1.)
- [base_trainer.py:281-282](src/trainers/base_trainer.py#L281-L282): the second string lacks its `f` prefix and a space, so it prints a literal `{tboard_port}`. (Moot: deleted in Phase 2.)
- `eval_one_epoch` is annotated as returning lists but returns numpy arrays. (Done in Phase 1.)
- [config_parser.py:122](src/config_parser.py#L122): `k.rsplit('.')[0]` should be `k.split('.', 1)[0]`. (Done in Phase 4.)
- [server.py:167,193](server/server.py#L167): `print(traceback.print_exc())` prints `None`.
- [base_trainer.py:91](src/trainers/base_trainer.py#L91) calls `len(gpu_dev)` when `gpu_device` may be an int in YAML. (Moot: 4.4 removed `gpu_device`.)
- [base_trainer.py:233-236](src/trainers/base_trainer.py#L233-L236): the verbose block reads `test_set.data` even when there is no test set. (Done in Phase 4.)
- `torch.jit.load` is called without `map_location`. (Done in Phase 1.)
- `torchvision.transforms.transforms` (a private module) is imported in base_trainer.py:18, classifier_dataset.py:8 and common.py:15. (Done in Phase 1.)
- `zero_grad(set_to_none=True)` is the default since torch 2.0. (Done in Phase 4.)
- Use `torch.inference_mode()` instead of `torch.no_grad()` in eval and inference. (Done in Phase 4 in the trainer; export and the server change in Phase 6.)
- `find_latest_file_in_dir` uses ctime; `max(Path(d).glob("*.pth"), key=os.path.getmtime)` does it in one line. (Moot: deleted in Phase 1.)
- [generate_classmap_from_dataset.py](scripts/generate_classmap_from_dataset.py): the docstring says "convert dataset to tar format" and the function is named `generate_tar`.
- [duplicate_data.py:78-83](scripts/duplicate_data.py#L78-L83) copies N+1 files, and its main function is named `split_train_test`.
- [export_utils.py:140](src/utils/export_utils.py#L140): a comment contradicts the code below it.
- The export tests expect a `.pt` suffix while the code writes `.ptc`. (Moot: deleted in Phase 2.)
- The `comprehensive_clsf_metrics` docstring promises tp/tn/fp/fn keys it doesn't return. (Moot: deleted in Phase 2.)
- `ClassifierModel` (all three done in Phase 1 by the 1.7 rewrite):
  - Its docstring documents a `pretrained` parameter that doesn't exist.
  - `print("custom backbone")` silently ignores `num_classes`.
  - `num_classes > 0 and num_classes is not None` checks in the wrong order.
- `_get_webdataset_len` leaves tar handles open (ResourceWarning). (Moot: deleted in Phase 5.)
- [inference.py:24](inference.py#L24): the default run id is prefixed `test_` (copied from test.py). (Done in Phase 4.)
- [mnist_dataset.py:33](src/datasets/mnist_dataset.py#L33) raises a bare `Exception`. (Done in Phase 2.)
- `InputModel.back_url: str = None` should be `str | None`.
- [cleanup.sh](scripts/cleanup.sh) deletes `experiments*` and `logs*`, but runs are saved under `checkpoints/`.
- `.dockerignore` is padded with entries for Django, PyInstaller and Travis that don't apply here.

---

## Decisions

| Decision | Status | Item |
| --- | --- | --- |
| Dependency tool: uv or Poetry | **Decided: uv.** One lockfile, CPU/CUDA extras, faster CI and Docker, and three requirements files plus a script deleted. | 3.3 |
| Keep WebDataset? | **Decided: keep it,** and fix it per 5.5 (done in Phase 5). | 5.5 |
| Keep the server? | **Decided: keep it,** rebuilt as the minimal ONNX Runtime app. | 6.4 |
| Old checkpoints | **Decided: they don't need to load.** 1.7 and 4.1 rename the classifier head's state-dict keys; no key map. 4.2's full-state checkpoints don't load bare state dicts either. | 4.1 |
| Multi-GPU? | Open. Recommendation: single device; add DDP with `torchrun` if and when it's needed. Phase 1 already removed DataParallel (1.5). | 4.4 |
| Flatten the layout? | Open. Recommendation: yes, but last. | 7.11 |

## Considered and rejected

- **Lightning, Accelerate, Hydra or timm as core dependencies:** the training loop is about 100 lines, and these add more concepts than they remove. Accelerate is the first to revisit if DDP is ever needed; timm only if you need its model zoo.
- **W&B or MLflow:** TensorBoard ships with torch; external trackers add accounts and servers.
- **`transforms.v2` tensor pipeline for speed:** measured slower on CPU (v2 with `ToImage` first: 251 vs 540 img/s).
- **GPU JPEG decode, DALI or FFCV:** too much machinery for a template.
- **Hand-written `channels_last`, CUDA graphs or fused optimizers:** small or fragile gains for a CNN template; inductor handles memory layout under compile.
- **Server dynamic batching or async model workers:** sync `def` endpoints are enough.
- **torchao PT2E quantization:** a new dependency, when ONNX Runtime's `quantize_static` already works with an existing one.
- **EMA, MixUp/CutMix, gradient clipping, early stopping:** add per project when needed.
- **Caching dataset mean/std to a file:** moot once normalization comes from the pretrained weights.
- **Config inheritance (Hydra-style defaults lists):** four small YAML files don't need it.

---

## Execution order

One PR per phase. Phase 1 can be split if it gets large.

| PR | Contents | Risk | How to verify |
| --- | --- | --- | --- |
| 1 | Phase 1 fixes, each with one regression test | Low | pytest, plus an MNIST smoke run |
| 2 | Phase 2 deletions, with their tests | Low (main path unchanged) | pytest; `ruff check` (F401 catches leftover imports) |
| 3 | Phase 3: Python and dependency bump, lockfile and tooling, CI, Docker, the one-off `ruff format` commit | Medium | CI green on 3.11 and 3.14; `docker build` |
| 4 | Phase 4 training procedure | Medium | A 2-epoch run; resuming from `last.pth` continues at epoch 3 with the same LR |
| 5 | Phase 5 data pipeline and speed | Medium | Epoch time before and after; label counts unchanged |
| 6 | Phase 6 export, inference and server | Medium | Export parity check; `curl` the `/predict` endpoint |
| 7 | Phase 7 structure, plus nitpicks in files being touched | Medium–high (large diff) | Full suite plus smoke runs |

The smoke run works on CPU: `python train.py --cfg configs/mnist_config.yaml -o trainer.epochs=2`.

---

## Appendix A: test-suite results

| Environment | Result |
| --- | --- |
| Exact `poetry.lock` (Python 3.12, torch 2.8.0) | 119 passed, 1 failed\*, 21 skipped (CUDA-only) |
| `pip install -r requirements/train.txt` (transitive deps float) | 116 passed, 4 failed. Three are ONNX dynamo export failures caused by onnxscript 0.4.0 with onnx-ir 1.1. |
| Latest (Python 3.13, torch 2.14.1, torchvision 0.29.1, …) | 119 passed, 1 failed\* |
| Latest on Python 3.14.6 | 119 passed, 1 failed\*. Took 102 s vs 27 s because of forkserver with non-persistent workers. |

\* `test_get_git_revision_hash` failed only because the review ran from a non-git copy; it passes in the repo.

A smoke test (train 2 epochs → test → all 4 export modes) passes on both the lock and the latest stack, and the ONNX outputs match PyTorch. That smoke test used the classifier config on mock images. The MNIST config crashed at startup until Phase 1 (1.6e).

## Appendix B: dependencies

"Latest" is from PyPI on 2026-10-27; dates are 2026 unless noted. Exact versions belong in the lockfile. Phase 3 locked these ranges in `uv.lock` on 2026-10-28; pytest 9.2.0 and ruff 0.17.3 had been released by then.

| Package | Locked | Latest | Notes for this repo | Suggested range |
| --- | --- | --- | --- | --- |
| torch | 2.8.0 | 2.14.1 (Sep 30) | ONNX export defaults to dynamo since 2.9; TorchScript FutureWarnings; `torch.ao.quantization` deprecated | `>=2.14,<2.15` |
| torchvision | 0.23.0 | 0.29.1 | Needs torch 2.14. v1 transforms and string weight names still work. The current `torch <3` range is misleading, because `torchvision <0.24` pins torch to 2.8. | `>=0.29,<0.30` |
| onnxscript | 0.4.0 | 0.7.2 | 0.4.0 with onnx-ir 1.x breaks dynamo export ✅ | `>=0.7,<0.8` |
| onnxruntime | 1.21.1 | 1.31.1 (Oct 26) | Needs Python ≥3.11 | `>=1.31,<1.32` |
| onnx | 1.17.0 | 1.23.2 | Already transitive through onnxscript; keep it direct only if you call `onnx.checker` | transitive |
| webdataset | 0.2.100 | 1.0.2 (2025-06) | `shardshuffle=True`/None and `with_length` warn; `with_length` can't be pickled for spawn or forkserver workers | `>=1.0,<1.1` (or remove) |
| omegaconf | 2.3.0 | 2.4.0 (Oct 9) | Drops the antlr4 4.9.3 pin; current code works | `>=2.4,<2.5` |
| scikit-learn | 1.6.1 | 1.9.1 | Needs Python ≥3.11 | `>=1.9,<2` |
| tensorboard | 2.18.0 | 2.21.0 | No issues | `<3` |
| matplotlib | 3.10.0 | 3.11.2 | Needs Python ≥3.11 | `<4` |
| tqdm | 4.67.1 | 4.70.1 | No issues | `<5` |
| imageio | 2.37.0 | 2.38.1 | Only used by dead code and two scripts, which can use PIL | remove |
| torchinfo | 1.8.0 | 1.8.0 | Only used by the unused `print_summary` | remove |
| six, packaging | 1.17.0, 24.2 | n/a | Never imported | remove |
| urllib3, fonttools, filelock | security floors | 2.8.0, 4.67.0, 4.2.4 | The `filelock <4` cap blocks 4.x | floors only, or rely on the lock |
| fastapi | 0.129.0 | 0.162.0 | Routes work on latest | `>=0.162,<0.163` (server group) |
| uvicorn, python-multipart | 0.35.0, 0.0.22 | 0.54.0, 0.0.32 | No issues | latest minor (server group) |
| requests | 2.32.4 | 2.34.2 | Server only | remove after 6.4 |
| opencv-python, aiofiles | 4.12, 24.1 | 5.0, 25.1 | Unused | remove |
| pytest (+cov, mock) | 8.4.1 | 9.1.1 | `PytestRemovedIn10Warning` | `>=9.1,<10`; drop pytest-order (7.9) |

**Python:** `>=3.11,<3.15`.

| Version | End of life |
| --- | --- |
| 3.10 | October 2026 (passed) |
| 3.11 | October 2027 |
| 3.12 | October 2028 |
| 3.13 | October 2029 |
| 3.14 | October 2030 |

## Appendix C: target layout (optional)

```text
train.py  test.py  export.py  inference.py   # thin CLIs: parse args → load_config → call functions
configs/                                      # mnist.yaml, classifier.yaml (+ webdataset variant)
src/
  config.py     load_config(), make_run_dir(), seed_everything()
  data.py       build_transforms(), build_datasets(), build_dataloaders()
  models.py     MnistModel, CustomModel, torchvision_classifier()
  trainer.py    Trainer: fit(), validate(), evaluate(), save(), load()
  metrics.py    METRICS dict, plot_curves()
  export.py     export_onnx(), check_onnx_parity()
server/app.py   FastAPI + onnxruntime, one /predict endpoint
scripts/        data prep: split, duplicate, find corrupt, convert to shards
tests/          fast, offline unit tests + one smoke test
```
