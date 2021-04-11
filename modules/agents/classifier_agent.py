"""
Agent class for general classifier/feature extraction networks
"""
import os
import torch
from torch.backends import cudnn

from modules.agents.base_agent import BaseAgent
from modules.utils.statistics import print_cuda_statistics
from modules.utils.util import find_latest_file_in_dir


cudnn.benchmark = True


class ClassifierAgent(BaseAgent):

    def __init__(self, CONFIG):
        super().__init__(CONFIG)

        # define models
        self.model = self.CONFIG.ARCH.TYPE()

        # define dataset
        self.data_set = self.CONFIG.DATASET.TYPE

        # define train, validate, and test data_loader
        # in OSX systems DATALOADER.NUM_WORKERS should be set to 0 which might increasing training time
        self.train_data_loader = self.CONFIG.DATALOADER.TYPE(self.data_set.train_dataset,
                                                             batch_size=self.CONFIG.DATALOADER.BATCH_SIZE,
                                                             validation_split=self.CONFIG.DATALOADER.VALIDATION_SPLIT,
                                                             shuffle=self.CONFIG.DATALOADER.SHUFFLE,
                                                             num_workers=self.CONFIG.DATALOADER.NUM_WORKERS,
                                                             pin_memory=self.CONFIG.DATALOADER.PIN_MEMORY)
        self.test_data_loader = self.CONFIG.DATALOADER.TYPE(self.data_set.test_dataset,
                                                            batch_size=self.CONFIG.DATALOADER.BATCH_SIZE,
                                                            validation_split=0,
                                                            shuffle=self.CONFIG.DATALOADER.SHUFFLE,
                                                            num_workers=self.CONFIG.DATALOADER.NUM_WORKERS,
                                                            pin_memory=self.CONFIG.DATALOADER.PIN_MEMORY)
        # define loss and instantiate it
        self.loss = self.CONFIG.LOSS()

        # define optimizer
        self.optimizer = self.CONFIG.OPTIMIZER.TYPE(
            self.model.parameters(),
            lr=self.CONFIG.OPTIMIZER.LR,
            momentum=self.CONFIG.OPTIMIZER.MOMENTUM)

        # define lr scheduler
        self.scheduler = self.CONFIG.LR_SCHEDULER.TYPE(self.optimizer,
                                                       factor=self.CONFIG.LR_SCHEDULER.FACTOR,
                                                       patience=self.CONFIG.LR_SCHEDULER.PATIENCE)

        # initialize metrics dict
        self.best_metric_dict = {metric: [] for metric in self.CONFIG.METRICS}

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0

        # set cuda flag
        is_cuda = torch.cuda.is_available()
        if is_cuda and not self.CONFIG.USE_CUDA:
            self.logger.info(
                "WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = is_cuda & self.CONFIG.USE_CUDA

        # set the manual seed for torch
        self.manual_seed = self.CONFIG.SEED
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            torch.cuda.set_device(self.CONFIG.GPU_DEVICE[0])
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)
            self.device = torch.device("cuda")
            self.logger.info("Program will run on *****GPU-CUDA*****")
            print_cuda_statistics()
        else:
            torch.manual_seed(self.manual_seed)
            self.device = torch.device("cpu")
            self.logger.info("Program will run on *****CPU*****")

        # if training resume is True, load model from the latest checkpoint, if not found start from scratch.
        if self.CONFIG.TRAINER.RESUME:
            self.load_checkpoint(self.CONFIG.TRAINER.CHECKPOINT_DIR)
        else:
            print("Training will be done from scratch")

        # if using tensorboard, register graph for vis with dummy input
        if self.CONFIG.TRAINER.USE_TENSORBOARD:
            w = self.CONFIG.ARCH.INPUT_WIDTH
            h = self.CONFIG.ARCH.INPUT_HEIGHT
            c = self.CONFIG.ARCH.INPUT_CHANNEL
            _dummy_input = torch.ones(([1, c, h, w]))
            self.tboard_writer.add_graph(self.model, _dummy_input)

    def load_checkpoint(self, path):
        """
        Latest checkpoint loader from torch weights
        :param
            path: path to checkpoint file/folder with only weights,
                  if folder is used, latest checkpoint is loaded
        :return:
            None
        """
        ckpt_file = None
        if os.path.isfile(path):
            ckpt_file = path
        elif os.path.isdir(path):
            ckpt_file = find_latest_file_in_dir(path)

        if ckpt_file is None:
            msg = (f"'{path}' is not a torch weight file or a directory containing one. " +
                   "No weight were loaded and training will be done from scratch")
            self.logger.info(msg)
            return

        if self.cuda:  # if gpu is available
            self.model.load_state_dict(torch.load(ckpt_file))
        else:          # if gpu is not available
            self.model.load_state_dict(torch.load(ckpt_file,
                                                  map_location=torch.device('cpu')))
        self.logger.info(f"Loaded checkpoint {ckpt_file}")

    def save_checkpoint(self, filename="checkpoint.pth"):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :return:
        """
        # create checkpoint directory if it doesnt exist
        os.makedirs(self.CONFIG.TRAINER.CHECKPOINT_DIR, exist_ok=True)
        save_path = os.path.join(self.CONFIG.TRAINER.CHECKPOINT_DIR, filename)
        torch.save(self.model.state_dict(), save_path)

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
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

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        # set model to training mode
        self.model.train()
        cum_train_loss = 0
        train_size = 0
        correct = 0
        train_data_len = len(self.data_set.train_dataset) - (
            len(self.data_set.train_dataset) * self.CONFIG.DATALOADER.VALIDATION_SPLIT)

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            train_size += data.shape[0]
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
            self.tboard_writer.add_scalars('Loss (epoch)',
                                           {'train': train_loss},
                                           self.current_epoch)
            self.tboard_writer.add_scalars('Accuracy (epoch)',
                                           {'train': train_accuracy},
                                           self.current_epoch)

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        # set model to eval mode
        self.model.eval()
        cum_val_loss = 0
        val_size = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.train_data_loader.split_validation():
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
        self.scheduler.step(val_loss)

        self.logger.info('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, val_size, val_accuracy))
        if self.CONFIG.TRAINER.USE_TENSORBOARD:
            self.tboard_writer.add_scalars('Loss (epoch)',
                                           {'validation': val_loss},
                                           self.current_epoch)
            self.tboard_writer.add_scalars('Accuracy (epoch)',
                                           {'validation': val_accuracy},
                                           self.current_epoch)

    def export_as_onnx(self,
                       dummy_input,
                       onnx_save_path="checkpoints/onnx_model.onnx"):
        """
        ONNX format export function
        Model should use torch tensors & torch operators for proper export
        numpy values are treated as constant nodes by the tracing method
        args:
            dummy_input: dummy input for the model
            onnx_save_path: path where onnx file will be saved
        return:
        """
        # Export with ONNX
        torch.onnx.export(self.model,
                          dummy_input,
                          onnx_save_path,
                          export_params=True,
                          do_constant_folding=True,
                          opset_version=12,
                          input_names=['input'],    # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size', 1: "width", 2: "height"},    # variable length axes
                                        'output': {0: 'batch_size', 1: "width", 2: "height"}},
                          verbose=True)

    def finalize_exit(self):
        """
        operations for graceful exit
        """
        print("Graceful exit initiating")
