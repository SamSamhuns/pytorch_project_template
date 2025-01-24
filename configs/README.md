# Required Keys for Configuration YAML file

An example Configuration file is provided at `classifier_cpu_config.yaml`

## Sample YAML configuration file

```yaml
# Configuration for Image Classifier
# General settings
name: image_classifier  # Name of the project or task
save_dir: checkpoints  # Directory to save checkpoints and logs
git_hash: null  # Git hash for version control, set during runs
mode: TRAIN  # Mode: TRAIN or TEST
seed: 42  # Random seed for reproducibility
cudnn_deterministic: false  # Set to true for reproducible results (slower GPU performance if true)
cudnn_benchmark: true  # Enable to use faster GPU functions during training
device: cuda  # Device to run the model on (cuda or cpu)
gpu_device:  # List of GPU devices to use
- 0
use_amp: true  # Use automatic mixed precision for faster training and lower memory usage
torch_compile_model: false  # Compile the PyTorch model for optimized performance
quant_backend: null  # Quantization mode, used only in export

# Trainer configuration
trainer:
  type: ClassifierTrainer  # Trainer type (classification tasks)
  resume_checkpoint: null  # Path to checkpoint for resuming training
  save_best_only: true  # Save only the best checkpoint based on validation metrics
  batch_log_freq: 5  # Logging frequency within an epoch
  weight_save_freq: 5  # Frequency of saving model weights across epochs
  valid_freq: 2  # Frequency of validation during training
  epochs: 12  # Number of training epochs
  verbosity: 2  # Verbosity level of training logs
  use_tensorboard: true  # Enable TensorBoard logging
  tensorboard_port: null  # Port for TensorBoard (null for default)

# Model configuration
model:
  type: ClassifierModel  # Model type
  args:
    backbone: mobilenet_v2  # Backbone network for feature extraction
    feat_extract: false  # If true, freezes backbone layers and trains only the classifier head
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
  num_classes: 20  # Number of classes in the dataset
  preprocess:
    train_transform: ImagenetClassifierPreprocess  # Data preprocessing for training
    val_transform: ImagenetClassifierPreprocess  # Data preprocessing for validation
    test_transform: ImagenetClassifierPreprocess  # Data preprocessing for testing
    inference_transform: ImagenetClassifierPreprocess  # Data preprocessing for inference

# DataLoader configuration
dataloader:
  type: CustomDataLoader  # DataLoader type
  args:
    batch_size: 32  # Number of samples per batch
    num_workers: 4  # Number of worker processes for data loading
    shuffle: true  # Shuffle the dataset during training
    validation_split: 0.0  # Fraction of data to use for validation
    drop_last: false  # Drop the last incomplete batch
    pin_memory: true  # Enable pinned memory for faster data transfer to GPU
    prefetch_factor: 2  # Number of samples preloaded by each worker
    worker_init_fn: null  # Custom initialization function for workers

# Optimizer configuration
optimizer:
  type: SGD  # Optimizer type (Stochastic Gradient Descent)
  args:
    lr: 0.01  # Learning rate
    momentum: 0.9  # Momentum factor

# Loss function configuration
loss:
  type: NLLLoss  # Loss function type (Negative Log-Likelihood Loss)
  args: {}  # No additional arguments

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
  type: ReduceLROnPlateau  # Scheduler type (reduces LR when a metric stops improving)
  args:
    factor: 0.1  # Factor by which the learning rate is reduced
    patience: 8  # Number of epochs with no improvement before reducing LR
```