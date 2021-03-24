import os
import torch
import torch.nn.functional as F
from torch.backends import cudnn

from modules.agents.base_agent import BaseAgent
from modules.utils.statistics import print_cuda_statistics
from modules.utils.util import find_latest_file_in_dir


cudnn.benchmark = True


class MnistAgent(BaseAgent):

    def __init__(self, CONFIG):
        super().__init__(CONFIG)

        # define models
        self.model = self.CONFIG.ARCH.TYPE()

        # define dataset
        self.data_set = self.CONFIG.DATASET.TYPE

        # define train, validate, and test data_loader
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
        # define loss
        self.loss = self.CONFIG.LOSS

        # define optimizer
        self.optimizer = self.CONFIG.OPTIMIZER.TYPE(
            self.model.parameters(),
            lr=self.CONFIG.OPTIMIZER.LR,
            momentum=self.CONFIG.OPTIMIZER.MOMENTUM)

        # define lr scheduler
        self.scheduler = self.CONFIG.LR_SCHEDULER.TYPE(self.optimizer,
                                                       factor=self.CONFIG.LR_SCHEDULER.FACTOR,
                                                       patience=self.CONFIG.LR_SCHEDULER.PATIENCE)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        # initialize metrics dict
        self.best_metric_dict = {metric: [] for metric in self.CONFIG.METRICS}

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.CONFIG.USE_CUDA:
            self.logger.info(
                "WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.CONFIG.USE_CUDA

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

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.CONFIG.TRAINER.CHECKPOINT_DIR)
        # Summary Writer
        self.summary_writer = None

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
            self.model.load_state_dict(torch.load(path))
        else:          # if gpu is not available
            self.model.load_state_dict(torch.load(path,
                                                  map_location=torch.device('cpu')))
        self.logger.info(f"Loaded checkpoint {ckpt_file}")

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :return:
        """
        save_path = os.path.join(self.CONFIG.CHECKPOINT_DIR, filename)
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

                # save best ckpt
                for metric in self.CONFIG.METRICS:
                    mlist = self.best_metric_dict[metric]
                    if len(mlist) == 1 or mlist[-1] > mlist[-2]:
                        self.save_checkpoint(
                            filename=f"checkpoint_{epoch}.pth.tar")
            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        # set model to training mode
        self.model.train()
        train_data_len = len(self.data_set.train_dataset) - (
            len(self.data_set.train_dataset) * self.CONFIG.DATALOADER.VALIDATION_SPLIT)

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.CONFIG.TRAINER.LOG_FREQ == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch,
                    batch_idx * len(data),
                    train_data_len,
                    100. * batch_idx / len(self.train_data_loader),
                    loss.item()))
            self.current_iteration += 1

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
                cum_val_loss += F.nll_loss(output, target,
                                           size_average=False).item()
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

        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, val_size, val_accuracy))

    def finalize_exit(self):
        """
        operations for graceful exit
        """
        print("Graceful exit initiating")
