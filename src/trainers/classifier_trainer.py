"""
Time Series Classification trainer
Dataset should be in the format [batch_size, num_channels, seq_len]
"""
import time
import glob
import os.path as osp
from contextlib import nullcontext
from typing import Optional, List, Tuple

from PIL import Image
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

from src.trainers import BaseTrainer
from src.datasets.base_dataset import IMG_EXTENSIONS
from src.config_parser import ConfigParser
from src.losses import init_loss
from src.models import init_model
from src.metrics import calc_metric
from src.optimizers import init_optimizer
from src.schedulers import init_scheduler
from src.augmentations import init_transform
from src.utils.statistics import get_model_params
from src.utils.common import BColors


class ClassifierTrainer(BaseTrainer):
    """
    Main ClassifierTrainer with the train, test, validate, inference, load_checkpoint and save_checkpoint funcs
    """

    def __init__(self, config: ConfigParser, logger_name: str):
        super().__init__(config, logger_name)

        # ###### define model args for UCR Dataset ######
        model_args = self.config["model"]["args"]
        # warning when using wrong loss for binary classification
        if "c_out" in model_args and model_args["c_out"] == 2:
            if self.config["loss"]["type"] in {"NLLLoss", "CrossEntropyLoss"}:
                msg = f"{BColors.WARNING}WARNING: For useful confs. for two class outputs, " + \
                      f"BCE losses are preferable over {self.config['loss']['type']}{BColors.ENDC}"
                self.logger.warning("%s", msg)
        # define models
        self.model = init_model(
            self.config["model"]["type"], **model_args)

        # do not progress beyond for INFERENCE mode
        if config["mode"] in {"INFERENCE"}:
            self.model = self.model.to(self.device)
            if self.config.resume:
                self.model = self.load_checkpoint(self.model, self.config.resume)
            return

        # define loss and instantiate it
        self.loss = init_loss(
            self.config["loss"]["type"], **self.config["loss"]["args"])
        # define optimizer
        self.optimizer = init_optimizer(
            self.config["optimizer"]["type"], **self.config["optimizer"]["args"],
            params=self.model.parameters())
        # define lr scheduler
        self.scheduler = init_scheduler(
            self.config["lr_scheduler"]["type"], **self.config["lr_scheduler"]["args"],
            optimizer=self.optimizer)
        # initialize metrics dict
        self.best_val_metric_dict = {
            metric: [] for metric in self.config["metrics"]["val"]}
        # initialize counter
        self.current_epoch = 1
        self.current_iteration = 0

        # use multi-gpu if cuda available and multi-dev set with gpu_device
        gpu_device = self.config["gpu_device"]
        gpu_device = [gpu_device] if isinstance(gpu_device, int) else gpu_device
        if self.cuda and len(gpu_device) > 1 and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_device)

        # load model and loss to device
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        # handle param logging and weight init for DataParallel
        if isinstance(self.model, torch.nn.DataParallel):
            self.logger.info("Model info: %s", get_model_params(self.model.module))
            try:
                self.model.module.initialize_weights(method="he")
            except AttributeError as err:
                self.logger.warning("%s", err)
        else:
            self.logger.info("Model info: %s", get_model_params(self.model))
            try:
                self.model.initialize_weights(method="he")
            except AttributeError as err:
                self.logger.warning("%s", err)
        if self.config["torch_compile_model"]:
            self.logger.info("Using torch compile mode")
            self.model = torch.compile(self.model)

        # if --resume cli argument is provided, give precedence to --resume ckpt path
        if self.config.resume:
            self.model = self.load_checkpoint(
                self.model,  self.config.resume)
        # else use "resume_checkpoint" if provided in json config if --resume cli argument is absent
        elif self.config["trainer"]["resume_checkpoint"] is not None:
            self.model = self.load_checkpoint(
                self.model, self.config["trainer"]["resume_checkpoint"])
        # else load from scratch
        else:
            self.logger.info("Training will be done from scratch")

    def train_one_epoch(self) -> None:
        """
        One epoch of training
        """
        if self.config.verbose:
            self.logger.info("Training set:")
        epoch_start_tm = time.perf_counter()
        self.model.train()
        cum_train_loss = 0
        train_size = 0
        correct = 0
        train_data_len = len(self.data_set.train_set) - \
            (len(self.data_set.train_set)
             * self.config["dataloader"]["args"]["validation_split"])
        loss_type = self.config["loss"]["type"]

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            target = target.float() if loss_type in {"BCELoss", "BCEWithLogitsLoss"} else target
            self.optimizer.zero_grad(set_to_none=True)
            train_size += data.shape[0]

            # Enables autocasting for the forward pass (model + loss)
            with autocast() if self.config["use_amp"] else nullcontext():
                output = self.model(data).squeeze(-1)
                loss = self.loss(output, target)
            cum_train_loss += loss.item()
            # get the index of the max log-probability
            if loss_type in {"BCELoss", "BCEWithLogitsLoss"}:
                pred = (F.sigmoid(output) >= 0.5).to(int)
            else:
                pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss.backward()
            self.optimizer.step()
            # OneCycleLR scheduler step is called on each batch instead
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            if self.config.verbose and batch_idx % self.config["trainer"]["batch_log_freq"] == 0:
                self.logger.info(
                    "\tTrain Epoch: %s [%6d/%.0f (%.1f%%)] Loss: %.6f",
                    self.current_epoch,
                    batch_idx * len(data),
                    train_data_len,
                    100 * batch_idx / len(self.train_data_loader),
                    loss.item())
            if self.config["trainer"]["use_tensorboard"]:
                self.tboard_writer.add_scalar("Loss/train/iteration",
                                              loss.item(),
                                              self.current_iteration)
            self.current_iteration += 1

        train_loss = cum_train_loss / train_size
        train_accuracy = 100. * correct / train_size
        epoch_end_tm = time.perf_counter()
        epoch_tm = epoch_end_tm - epoch_start_tm
        if self.config.verbose:
            self.logger.info("\tEpoch time: %.2fs", epoch_tm)
            self.logger.info("\tAverage loss: %.4f, Accuracy: %s/%s (%.0f%%)\n",
                             train_loss, correct, train_size, train_accuracy)
        else:
            self.logger.info(
                "\tTrain Epoch %d, time: %.2fs, Loss: %.4f, Accuracy: %.2f%%",
                self.current_epoch, epoch_tm, train_loss, train_accuracy)

        # if no val_data_loader, then run ReduceLROnPlateau on train_loss instead
        if self.val_data_loader is None and isinstance(self.scheduler, ReduceLROnPlateau):
            # ReduceLROnPlateau scheduler takes metrics during its step call
            self.scheduler.step(metrics=train_loss)

        if self.config["trainer"]["use_tensorboard"]:
            self.tboard_writer.add_images('preprocessed image batch',
                                          next(iter(self.train_data_loader))[0],
                                          self.current_epoch)
            self.tboard_writer.add_scalar("Loss/train/epoch",
                                          train_loss,
                                          self.current_epoch)
            self.tboard_writer.add_scalar("Accuracy/train/epoch",
                                          train_accuracy,
                                          self.current_epoch)
            self.tboard_writer.add_scalar("Learning Rate/train/epoch",
                                          self.optimizer.param_groups[0]["lr"],
                                          self.current_epoch)

    def eval_one_epoch(self, data_loader: DataLoader) -> Tuple[float, List[int], List[float], List[int]]:
        """
        Evaluate model on one epoch with the given dataloader
        """
        self.model.eval()
        cumu_test_loss = 0
        loss_type = self.config["loss"]["type"]
        y_true, y_score, y_pred = [], [], []
        with torch.no_grad():
            for data, target in tqdm.tqdm(data_loader):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                target = target.float() if loss_type in {"BCELoss", "BCEWithLogitsLoss"} else target
                output = self.model(data).squeeze(-1)
                # sum up batch loss
                cumu_test_loss += self.loss(output, target).item()
                # get the log-probability (proba) & index of the max log-probability (preds)
                if loss_type in {"BCELoss", "BCEWithLogitsLoss"}:
                    proba = F.sigmoid(output)
                    pred = (proba >= 0.5).to(int)
                else:
                    proba, pred = F.softmax(output, dim=1).max(1)

                y_true.append(target)
                y_score.append(proba)
                y_pred.append(pred)
        y_true = torch.concat(y_true).cpu().numpy()
        y_score = torch.concat(y_score).cpu().numpy()
        y_pred = torch.concat(y_pred).cpu().numpy()

        return cumu_test_loss, y_true, y_score, y_pred

    def train(self) -> None:
        """
        Main training function with loop
        """
        train_start_tm = time.perf_counter()
        epochs = self.config["trainer"]["epochs"]
        self.logger.info("Starting training...")
        for epoch in range(1, epochs + 1):
            self.train_one_epoch()
            if epoch % self.config["trainer"]["valid_freq"] == 0:
                if self.val_data_loader is not None:
                    self.validate()
            # scheduler.step should be called after validate()
            if isinstance(self.scheduler, (ReduceLROnPlateau, OneCycleLR)):
                # ReduceLROnPlateau takes metrics during its step call (inside validate())
                # if no val, ReduceLROnPlateau takes train loss as metric (inside train_one_epoch())
                # OneCycleLR is called on each batch instead (inside train_one_epoch())
                pass
            else:
                self.scheduler.step()

            # save trained model checkpoint
            if self.config["trainer"]["save_best_only"]:
                for metric in self.config["metrics"]["val"]:
                    mlist = self.best_val_metric_dict[metric]
                    # save if 1 or 0 metrics present or if metric is better than previous best
                    if (len(mlist) <= 1 or
                        (metric == "loss" and mlist[-1][1] < mlist[-2][1]) or
                            (metric == "accuracy_score" and mlist[-1][1] > mlist[-2][1])):
                        self.save_checkpoint(self.model, "best.pth")
            elif epoch % self.config["trainer"]["weight_save_freq"] == 0:
                self.save_checkpoint(self.model, f"checkpoint_{epoch}.pth")
            self.current_epoch += 1
        train_end_tm = time.perf_counter()
        train_tm = train_end_tm - train_start_tm

        self.logger.info("Total train time: %.3fs", train_tm)
        self.logger.info("Average epoch time: %.3fs", train_tm / max(epochs, 1))
        self.logger.info("Finished training.\n")

    def validate(self) -> None:
        """
        One epoch of model validation
        """
        t_0 = time.perf_counter()

        cumu_val_loss, y_true, _, y_pred = self.eval_one_epoch(
            self.val_data_loader)

        if self.data_set.val_set:
            val_size = len(self.data_set.val_set)
        else:  # take val set size from sampler if data_set.val_set is None
            val_size = len(self.val_data_loader.sampler.indices)
        val_loss = cumu_val_loss / val_size
        correct = sum(y_true == y_pred)
        val_accuracy = 100. * correct / val_size

        # add metrics to tracking best_val_metric_dict
        if "accuracy_score" in self.best_val_metric_dict:
            self.best_val_metric_dict["accuracy_score"].append(
                [self.current_epoch, val_accuracy])
        if "loss" in self.best_val_metric_dict:
            self.best_val_metric_dict["loss"].append(
                [self.current_epoch, val_loss])

        if isinstance(self.scheduler, ReduceLROnPlateau):
            # ReduceLROnPlateau scheduler takes metrics during its step call
            self.scheduler.step(metrics=val_loss)

        t_1 = time.perf_counter()
        if self.config.verbose:
            self.logger.info("Validation set:")
            self.logger.info("\tVal time: %.2fs", (t_1 - t_0))
            self.logger.info("\tAverage loss: %.4f, Accuracy: %s/%s (%.0f%%)\n",
                             val_loss, correct, val_size, val_accuracy)
        else:
            self.logger.info(
                "\tValidation at Epoch %d, time: %.2fs, Loss: %.4f, Accuracy: %.2f%%",
                self.current_epoch, (t_1 - t_0), val_loss, val_accuracy)
        if self.config["trainer"]["use_tensorboard"]:
            self.tboard_writer.add_scalar("Loss/validation/epoch",
                                          val_loss,
                                          self.current_epoch)
            self.tboard_writer.add_scalar("Accuracy/validation/epoch",
                                          val_accuracy,
                                          self.current_epoch)

    def test(self, weight_path: Optional[str] = None) -> None:
        """
        main test function
        args:
            weight_path: Path to pth weight file that will be loaded for test
                         Default is set to None which uses latest chkpt weight file
        """
        self.logger.info("Testing model...\n")
        test_start_tm = time.perf_counter()

        if weight_path is not None:
            print(f"Loading new checkpoint from {weight_path} for testing")
            self.model = self.load_checkpoint(self.model, weight_path)
        if self.test_data_loader is None:
            raise NotImplementedError("test_data_loader is None or missing."
                                      "test_path might not have been set")

        cumu_test_loss, y_true, y_score, y_pred = self.eval_one_epoch(
            self.test_data_loader)

        test_size = len(self.data_set.test_set)
        test_loss = cumu_test_loss / test_size

        self.logger.info("Test set:")
        self.logger.info("\tAverage loss: %.4f", test_loss)
        if self.config["trainer"]["use_tensorboard"]:
            self.tboard_writer.add_scalar(
                "Loss/test/epoch", test_loss, self.current_epoch)
        # add all test metrics
        for metric_name in self.config["metrics"]["test"]:
            if metric_name == "roc_auc_score" and np.unique(y_true).size > 2:
                continue  # dont calculate roc_auc_score for mult class clsf
            metric_val = calc_metric(
                metric_name, y_true=y_true, y_score=y_score, y_pred=y_pred)
            if metric_name in {"confusion_matrix", "classification_report"}:
                self.logger.info("\t%s: \n%s", metric_name, metric_val)
            else:
                self.logger.info("\t%s: %.4f", metric_name, metric_val)
                if self.config["trainer"]["use_tensorboard"]:
                    self.tboard_writer.add_scalar(
                        f"{metric_name}/test/epoch", metric_val, self.current_epoch)

        test_end_tm = time.perf_counter()
        self.logger.info("\nTest time: %.3fs", test_end_tm - test_start_tm)
        self.logger.info("Finished testing model.")

    def inference(self, source_path: str, weight_path: Optional[str] = None, log_txt_preds: bool = True) -> List[Tuple[str, float]]:
        """
        Run inference on an image file or directory with existing model or model loaded with new weight_path
        Args:
            source_path: str
        """
        if weight_path is not None:
            print("Loading new checkpoint for inferencing")
            self.model = self.load_checkpoint(self.model, weight_path)
        self.model.eval()

        image_list = []
        if osp.isdir(source_path):
            fpaths = glob.glob(osp.join(source_path, "*"))
            image_list = [path for path in fpaths if osp.splitext(
                path)[-1] in IMG_EXTENSIONS]
        elif osp.isfile(source_path) and osp.splitext(source_path)[-1] in IMG_EXTENSIONS:
            image_list = [source_path]
        else:
            raise ValueError(
                f"Inference source {source_path} is not an image file or dir with images")

        pred_file_labels = []
        inference_transform = init_transform(
            self.config["dataset"]["preprocess"]["inference_transform"]).inference
        with torch.no_grad():
            for image_path in image_list:
                image = Image.open(image_path).convert('RGB')
                data = inference_transform(image)
                data = data.to(self.device).unsqueeze(0)
                output = self.model(data)
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1].item()

                self.logger.info(
                    "Image path=%s: predicted label=%s", image_path, pred)
                pred_file_labels.append([image_path, pred])
        if log_txt_preds:
            with open(osp.join(self.config.log_dir, "pred.txt"), 'w', encoding="utf-8") as pred_ptr:
                for image_path, pred in pred_file_labels:
                    pred_ptr.write(f"{image_path}, {pred}\n")

        self.logger.info("Inference complete for %s", source_path)
        return pred_file_labels
