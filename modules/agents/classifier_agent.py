"""
Agent class for general classifier/feature extraction networks
"""
from contextlib import nullcontext
import os

import torch
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.agents.base_agent import BaseAgent
from modules.utils.statistics import print_cuda_statistics
from modules.utils.util import find_latest_file_in_dir


class ClassifierAgent(BaseAgent):
    """
    Main ClassifierAgent with the train, test, validate, load_checkpoint and save_checkpoint funcs
    """

    def __init__(self, CONFIG):
        """
        args:
            CONFIG: a config.py file loaded as a pymodule with importlib
        """
        super().__init__(CONFIG)

        # define models
        self.model = self.CONFIG.ARCH.TYPE(**self.CONFIG.ARCH.ARGS,
                                           num_classes=self.CONFIG.DATASET.NUM_CLASSES)
        # define dataset
        self.data_set = self.CONFIG.DATASET.TYPE(**{**self.CONFIG.DATASET.DATA_DIR,
                                                    **self.CONFIG.DATASET.PREPROCESS})
        # define train, validate, and test data_loaders
        self.train_data_loader, self.val_data_loader, self.test_data_loader = None, None, None
        # in OSX systems DATALOADER.NUM_WORKERS should be set to 0 which might increasing training time
        self.train_data_loader = self.CONFIG.DATALOADER.TYPE(self.data_set.train_dataset,
                                                             **self.CONFIG.DATALOADER.ARGS)
        # if VAL_DIR is not None and VALIDATION_SPLIT must be 0
        # if no val dir is provided, take val split from training data
        if self.CONFIG.DATASET.DATA_DIR.val_dir is None:
            self.val_data_loader = self.train_data_loader.split_validation()
        # if val dir is provided, use all data inside val dir for validation
        elif self.CONFIG.DATASET.DATA_DIR.val_dir is not None:
            self.val_data_loader = self.CONFIG.DATALOADER.TYPE(self.data_set.val_dataset,
                                                               **self.CONFIG.DATALOADER.ARGS)
        if self.CONFIG.DATASET.DATA_DIR.test_dir is not None:
            self.test_data_loader = self.CONFIG.DATALOADER.TYPE(self.data_set.test_dataset,
                                                                **dict(self.CONFIG.DATALOADER.ARGS,
                                                                       validation_split=0))
        # define loss and instantiate it
        self.loss = self.CONFIG.LOSS()
        # define optimizer
        self.optimizer = self.CONFIG.OPTIMIZER.TYPE(self.model.parameters(),
                                                    **self.CONFIG.OPTIMIZER.ARGS)
        # define lr scheduler
        self.scheduler = self.CONFIG.LR_SCHEDULER.TYPE(self.optimizer,
                                                       **self.CONFIG.LR_SCHEDULER.ARGS)
        # initialize metrics dict
        self.best_metric_dict = {metric: [] for metric in self.CONFIG.METRICS}
        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        # if using tensorboard, register graph for vis with dummy input
        if self.CONFIG.TRAINER.USE_TENSORBOARD:
            w = self.CONFIG.ARCH.INPUT_WIDTH
            h = self.CONFIG.ARCH.INPUT_HEIGHT
            c = self.CONFIG.ARCH.INPUT_CHANNEL
            _dummy_input = torch.ones(
                [1, c, h, w], dtype=torch.float32)
            self.tboard_writer.add_graph(self.model, _dummy_input)
        # set cuda flag
        is_cuda = torch.cuda.is_available()
        gpu_device = self.CONFIG.GPU_DEVICE
        if is_cuda and not self.CONFIG.USE_CUDA:
            self.logger.info(
                "WARNING: CUDA device is available, enable CUDA for faster training")
        # set cuda devices if available or use cpu
        self.cuda = is_cuda & self.CONFIG.USE_CUDA
        self.manual_seed = self.CONFIG.SEED
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            torch.backends.cudnn.deterministic = self.CONFIG.CUDNN_DETERMINISTIC
            torch.backends.cudnn.benchmark = self.CONFIG.CUDNN_BENCHMARK
            if len(gpu_device) > 1 and torch.cuda.device_count() > 1:
                # use multi-gpu devices from config GPU_DEVICE
                self.model = torch.nn.DataParallel(
                    self.model, device_ids=gpu_device)
            else:
                # use one cuda gpu device from config GPU_DEVICE
                torch.cuda.set_device(gpu_device[0])
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)
            self.logger.info(f"Program will run on GPU device {self.device}")
            print_cuda_statistics()
        else:
            torch.manual_seed(self.manual_seed)
            self.device = torch.device("cpu")
            self.logger.info("Program will run on CPU")
        # use autiomatic mixed precision if set in config
        self.use_amp = self.CONFIG.USE_AMP
        # if training resume is True, load model from the latest checkpoint, if not found start from scratch.
        if self.CONFIG.TRAINER.RESUME:
            self.load_checkpoint(self.CONFIG.TRAINER.CHECKPOINT_DIR)
        else:
            print("Training will be done from scratch")

    def load_checkpoint(self, path) -> None:
        """
        Latest checkpoint loader from torch weights
        args:
            path: path to checkpoint file/folder with only weights,
                  if folder is used, latest checkpoint is loaded
        """
        ckpt_file = None
        if os.path.isfile(path):
            ckpt_file = path
        elif os.path.isdir(path):
            ckpt_file = find_latest_file_in_dir(path)

        if ckpt_file is None:
            msg = (f"'{path}' is not a torch weight file or a directory containing one. " +
                   "No weights were loaded and training will be done from scratch")
            self.logger.info(msg)
            return

        if self.cuda:  # if gpu is available
            self.model.load_state_dict(torch.load(ckpt_file))
        else:          # if gpu is not available
            self.model.load_state_dict(torch.load(ckpt_file,
                                                  map_location=torch.device('cpu')))
        self.logger.info(f"Loaded checkpoint {ckpt_file}")

    def save_checkpoint(self, filename="checkpoint.pth") -> None:
        """
        Checkpoint saver
        args:
            file_name: name of the checkpoint file
        """
        # create checkpoint directory if it doesnt exist
        os.makedirs(self.CONFIG.TRAINER.CHECKPOINT_DIR, exist_ok=True)
        save_path = os.path.join(self.CONFIG.TRAINER.CHECKPOINT_DIR, filename)
        torch.save(self.model.state_dict(), save_path)

    def train(self) -> None:
        """
        Main training function with loop
        """
        for epoch in range(self.CONFIG.TRAINER.EPOCHS):
            self.train_one_epoch()
            if epoch % self.CONFIG.TRAINER.VALID_FREQ:
                self.validate()

                # save trained model checkpoint
                for metric in self.CONFIG.METRICS:
                    mlist = self.best_metric_dict[metric]
                    if (not self.CONFIG.TRAINER.SAVE_BEST_ONLY or
                            (len(mlist) == 1 or mlist[-1] > mlist[-2])):
                        self.save_checkpoint(
                            filename=f"checkpoint_{epoch}.pth")
            self.current_epoch += 1

    def train_one_epoch(self) -> None:
        """
        One epoch of training
        """
        self.model.train()
        cum_train_loss = 0
        train_size = 0
        correct = 0
        train_data_len = len(self.data_set.train_dataset) - (
            len(self.data_set.train_dataset) * self.CONFIG.DATALOADER.ARGS.validation_split)

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            train_size += data.shape[0]

            # Enables autocasting for the forward pass (model + loss)
            with autocast() if self.use_amp else nullcontext():
                output = self.model(data)
                loss = self.loss(output, target)
            cum_train_loss += loss.item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss.backward()
            self.optimizer.step()
            if batch_idx % self.CONFIG.TRAINER.LOG_FREQ == 0:
                self.logger.info('Train Epoch: {} [{:6d}/{:.0f} ({:.1f}%)] Loss: {:.6f}'.format(
                    self.current_epoch,
                    batch_idx * len(data),
                    train_data_len,
                    100 * batch_idx / len(self.train_data_loader),
                    loss.item()))
            if self.CONFIG.TRAINER.USE_TENSORBOARD:
                self.tboard_writer.add_scalars('Loss (iteration)',
                                               {'train': loss.item()},
                                               self.current_iteration)
            self.current_iteration += 1

        train_loss = cum_train_loss / train_size
        train_accuracy = 100. * correct / train_size
        self.logger.info('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, train_size, train_accuracy))

        if self.CONFIG.TRAINER.USE_TENSORBOARD:
            self.tboard_writer.add_images('preprocessed image batch',
                                          next(iter(self.train_data_loader))[
                                              0],
                                          self.current_epoch)
            self.tboard_writer.add_scalars('Loss (epoch)',
                                           {'train': train_loss},
                                           self.current_epoch)
            self.tboard_writer.add_scalars('Accuracy (epoch)',
                                           {'train': train_accuracy},
                                           self.current_epoch)

    def validate(self) -> None:
        """
        One cycle of model validation
        """
        # set model to eval mode
        self.model.eval()
        cum_val_loss = 0
        val_size = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.val_data_loader:
                data, target = data.to(self.device), target.to(self.device)
                val_size += data.shape[0]
                output = self.model(data)
                # sum up batch loss
                cum_val_loss += self.loss(output, target).item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss = cum_val_loss / val_size
        val_accuracy = 100. * correct / val_size

        # add metrics to tracking best_metric_dicts
        if "val_accuracy" in self.best_metric_dict:
            self.best_metric_dict["val_accuracy"].append(
                [val_accuracy, self.current_epoch])
        if "val_loss" in self.best_metric_dict:
            self.best_metric_dict["val_loss"].append(
                [val_accuracy, self.current_epoch])

        # scheduler.step should be called after validate()
        # ReduceLROnPlateau scheduler takes metrics during its step call
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metrics=val_loss)
        else:
            self.scheduler.step()

        self.logger.info('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, val_size, val_accuracy))
        if self.CONFIG.TRAINER.USE_TENSORBOARD:
            self.tboard_writer.add_scalars('Loss (epoch)',
                                           {'validation': val_loss},
                                           self.current_epoch)
            self.tboard_writer.add_scalars('Accuracy (epoch)',
                                           {'validation': val_accuracy},
                                           self.current_epoch)

    def test(self, weight_path=None) -> None:
        """
        main test function
        args:
            weight_path: Path to pth weight file that will be loaded for test
                         Default is set to None which uses latest chkpt weight file
        """
        if weight_path is not None:
            print("Loading new checkpoint for testing")
            self.load_checkpoint(weight_path)
        if self.test_data_loader is None:
            raise NotImplementedError("test_data_loader is missing. " +
                                      "test_dir might not have been set")
        # set model to eval mode
        self.model.eval()
        cum_test_loss = 0
        test_size = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_data_loader:
                data, target = data.to(self.device), target.to(self.device)
                test_size += data.shape[0]
                output = self.model(data)
                # sum up batch loss
                cum_test_loss += self.loss(output, target).item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss = cum_test_loss / test_size
        test_accuracy = 100. * correct / test_size

        self.logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, test_size, test_accuracy))
        if self.CONFIG.TRAINER.USE_TENSORBOARD:
            self.tboard_writer.add_scalars('Test Loss (epoch)',
                                           {'Test': test_loss},
                                           self.current_epoch)
            self.tboard_writer.add_scalars('Test Accuracy (epoch)',
                                           {'Test': test_accuracy},
                                           self.current_epoch)

    def export_as_onnx(self,
                       dummy_input,
                       onnx_save_path="checkpoints/onnx_model.onnx") -> None:
        """
        ONNX format export function
        Model should use torch tensors & torch operators for proper export
        numpy values are treated as constant nodes by the tracing method
        args:
            dummy_input: dummy input for the model
            onnx_save_path: path where onnx file will be saved
        """
        # Export with ONNX
        torch.onnx.export(self.model,
                          dummy_input,
                          onnx_save_path,
                          export_params=True,
                          do_constant_folding=True,
                          opset_version=11,
                          input_names=['input'],    # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size', 2: "height", 3: "width"},    # variable length axes
                                        'output': {0: 'batch_size', 2: "height", 3: "width"}},
                          verbose=False)

    def finalize_exit(self):
        """
        operations for graceful exit
        """
        print("Graceful exit initiating")
