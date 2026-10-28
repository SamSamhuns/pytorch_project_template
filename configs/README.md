# Required Keys for Configuration YAML file

An example Configuration file is provided at `classifier_cpu_config.yaml`

## Sample YAML configuration file

```yaml
# Configuration for Image Classifier
# General settings
experiment_name: image_classifier  # Name of experiment, name of the dir for saving logs, metrics & models
save_dir: checkpoints  # Directory to save checkpoints and logs
git_hash: null  # Git hash for version control, set during runs
seed: 42  # Random seed for reproducibility
reproducible: false  # true trades speed for determinism: no cuDNN benchmarking, deterministic algorithms where they exist
device: cuda  # Device to run on: cpu, cuda or cuda:1. Falls back to cpu when CUDA is unavailable
use_amp: true  # bfloat16 mixed precision for training and evaluation; needs an Ampere or newer GPU, often slower on CPU
torch_compile_model: false  # Compile the PyTorch model for optimized performance

# Trainer configuration
trainer:
  type: ClassifierTrainer  # Trainer type (classification tasks)
  resume_checkpoint: null  # Checkpoint to resume from, e.g. .../models/last.pth; restores the optimizer, LR schedule and epoch too
  batch_log_freq: 5  # Logging frequency within an epoch
  valid_freq: 2  # Validate every N epochs; best.pth is saved when val accuracy improves, last.pth every epoch
  epochs: 12  # Number of training epochs
  verbosity: 2  # Verbosity level of training logs
  use_tensorboard: true  # Enable TensorBoard logging

# Model configuration
model:
  type: ClassifierModel  # Model type
  args:
    backbone: mobilenet_v2  # Backbone network for feature extraction
    feat_extract: false  # If true, the last Linear layer becomes Identity and the model returns features, not class scores
    num_classes: 20  # Number of output classes
    pretrained_weights: MobileNet_V2_Weights.IMAGENET1K_V1  # Pretrained weights to initialize the backbone
  info:
    input_width: 224  # Input image width
    input_height: 224  # Input image height
    input_channel: 3  # Number of input channels (e.g., RGB images)

# Dataset configuration
dataset:
  type: ClassifierDataset  # Dataset type
  args:
    root: data/birds_dataset  # Root directory of the dataset
    train_path: train  # Subdirectory for training data
    val_path: valid  # Subdirectory for validation data
    test_path: test  # Subdirectory for test data
    validation_split: 0.0  # Without a val_path, the fraction of the train data to validate on
  num_classes: 20  # Number of classes in the dataset
  preprocess:
    type: ImagenetClassifierPreprocess  # Random crops and flips for training; one eval transform for val, test and inference
    args:
      weights: ${model.args.pretrained_weights}  # The eval resize, crop and normalization come from these weights

# DataLoader configuration
dataloader:  # torch DataLoaders; the train loader shuffles, the val and test loaders don't
  batch_size: 32  # Number of samples per batch
  num_workers: 4  # Number of worker processes for data loading
  pin_memory: true  # Enable pinned memory for faster data transfer to GPU

# Optimizer configuration
optimizer:
  type: SGD  # Optimizer type (Stochastic Gradient Descent)
  args:
    lr: 0.01  # Learning rate
    momentum: 0.9  # Momentum factor

# Loss function configuration
loss:
  type: CrossEntropyLoss  # The models output logits, which CrossEntropyLoss expects
  args:
    label_smoothing: 0.1  # Optional; softens the one-hot targets, as torchvision's training recipe does

# Metrics configuration
metrics:
  val:
  - accuracy_score  # Validation metric: accuracy
  - f1_score  # Validation metric: F1 score
  test:
  - accuracy_score  # Test metric: accuracy
  - f1_score  # Test metric: F1 score
  - precision_score  # Test metric: precision
  - recall_score  # Test metric: recall
  - confusion_matrix  # Test metric: confusion matrix
  - classification_report  # Test metric: detailed classification report

# Learning rate scheduler configuration
lr_scheduler:
  type: CosineAnnealingLR  # Any scheduler in torch.optim.lr_scheduler, stepped once per epoch
  args:
    T_max: ${trainer.epochs}  # Epochs to decay the LR over; ${trainer.epochs} copies that value
  warmup_epochs: 0  # Epochs of linear warmup from 10% of the LR; not supported with ReduceLROnPlateau
```
