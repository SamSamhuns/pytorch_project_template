"""
Agent class for general classifier/feature extraction networks
"""
import time
import glob
import os.path as osp
from functools import partial
from contextlib import nullcontext
from typing import Optional, List, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

from modules.config_parser import ConfigParser
from modules.agents.base_agent import BaseAgent
from modules.datasets.base_dataset import IMG_EXTENSIONS
from modules.utils.util import rgetattr

import modules.losses as module_losses
import modules.models as module_models
import modules.metrics as module_metrics
import modules.optimizers as module_optimizers
import modules.schedulers as module_lr_schedulers
import modules.augmentations as module_transforms


class ClassifierAgent(BaseAgent):
    """
    Main ClassifierAgent with the train, test, validate, load_checkpoint and save_checkpoint funcs
    """

    def __init__(self, config: ConfigParser, logger_name: str):
        super().__init__(config, logger_name)

        num_classes = self.config["dataset"]["num_classes"]
        # define models
        backbone = self.config["arch"]["backbone"]
        self.model = self.config.init_obj(
            "arch", module_models,
            backbone=partial(getattr(module_models, backbone)
                             ) if backbone else None,
            num_classes=num_classes)

        # use multi-gpu if cuda available and multi-dev set with gpu_device
        gpu_device = self.config["gpu_device"]
        gpu_device = [gpu_device] if isinstance(gpu_device, int) else gpu_device
        if self.cuda and len(gpu_device) > 1 and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_device)

        # do not progress beyond for INFERENCE mode
        if config["mode"] in {"INFERENCE"}:
            self.model = self.model.to(self.device)
            if self.config.resume:
                self.model = self.load_checkpoint(self.model, self.config.resume)
            return

        # define loss and instantiate it
        self.loss = self.config.init_obj("loss", module_losses)
        # define optimizer
        self.optimizer = self.config.init_obj("optimizer", module_optimizers,
                                              params=self.model.parameters())
        # define lr scheduler
        self.scheduler = self.config.init_obj("lr_scheduler", module_lr_schedulers,
                                              optimizer=self.optimizer)
        # initialize metrics dict
        self.best_val_metric_dict = {
            metric: [] for metric in self.config["metrics"]["val"]}
        # initialize counter
        self.current_epoch = 1
        self.current_iteration = 0
        # if using tensorboard, register graph for vis with dummy input
        if self.config["trainer"]["use_tensorboard"]:
            i_w = self.config["arch"]["input_width"]
            i_h = self.config["arch"]["input_height"]
            i_c = self.config["arch"]["input_channel"]
            _dummy_input = torch.ones(
                [1, i_c, i_h, i_w], dtype=torch.float32)
            self.tboard_writer.add_graph(self.model, _dummy_input)

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        if self.config["torch_compile_model"]:
            self.logger.info("Using torch compile mode")
            self.model = torch.compile(self.model)

        # if --resume cli argument is provided, give precedence to --resume ckpt path
        if self.config.resume:
            self.model = self.load_checkpoint(self.model, self.config.resume)
        # else use 'resume_checkpoint' if provided in json config if --resume cli argument is absent
        elif self.config["trainer"]["resume_checkpoint"] is not None:
            self.model = self.load_checkpoint(self.model, self.config["trainer"]["resume_checkpoint"])
        # else load from scratch
        else:
            self.logger.info("Training will be done from scratch")

    def train(self) -> None:
        """
        Main training function with loop
        """
        t0 = time.perf_counter()
        for epoch in range(1, self.config["trainer"]["epochs"] + 1):
            self.train_one_epoch()
            if epoch % self.config["trainer"]["valid_freq"] == 0:
                if self.val_data_loader is not None:
                    self.validate()
            # scheduler.step should be called after validate()
            if isinstance(self.scheduler, (ReduceLROnPlateau, OneCycleLR)):
                # ReduceLROnPlateau takes metrics during its step call (inside validate())
                # if no validation, ReduceLROnPlateau takes train loss as metric (inside train_one_epoch())
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

            if (not self.config["trainer"]["save_best_only"] and
                    epoch % self.config["trainer"]["weight_save_freq"] == 0):
                self.save_checkpoint(self.model, f"checkpoint_{epoch}.pth")
            self.current_epoch += 1
        t1 = time.perf_counter()
        train_tm = t1 - t0

        self.logger.info("Total train time: %.3fs", train_tm)
        self.logger.info("Average epoch time: %.3fs", train_tm / self.config["trainer"]["epochs"])
        self.logger.info('Finished training.\n')

    def train_one_epoch(self) -> None:
        """
        One epoch of training
        """
        self.logger.info('Training set:')
        t_0 = time.perf_counter()
        self.model.train()
        cum_train_loss = 0
        train_size = 0
        correct = 0
        train_data_len = len(self.data_set.train_set) - (
            len(self.data_set.train_set) * self.config["dataloader"]["args"]["validation_split"])

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            train_size += data.shape[0]

            # Enables autocasting for the forward pass (model + loss)
            with autocast() if self.config["use_amp"] else nullcontext():
                output = self.model(data)
                loss = self.loss(output, target)
            cum_train_loss += loss.item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss.backward()
            self.optimizer.step()
            # OneCycleLR scheduler step is called on each batch instead
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            if batch_idx % self.config["trainer"]["batch_log_freq"] == 0:
                self.logger.info(
                    '\tTrain Epoch: %s [%6d/%.0f (%.1f%%)] Loss: %.6f',
                    self.current_epoch,
                    batch_idx * len(data),
                    train_data_len,
                    100 * batch_idx / len(self.train_data_loader),
                    loss.item())
            if self.config["trainer"]["use_tensorboard"]:
                self.tboard_writer.add_scalar('Loss/train/iteration',
                                              loss.item(),
                                              self.current_iteration)
            self.current_iteration += 1

        train_loss = cum_train_loss / train_size
        train_accuracy = 100. * correct / train_size
        t_1 = time.perf_counter()
        self.logger.info('\tEpoch time: %.2fs', (t_1 - t_0))
        self.logger.info('\tAverage loss: %.4f, Accuracy: %s/%s (%.0f%%)\n',
                         train_loss, correct, train_size, train_accuracy)
        # if no val_data_loader, then run ReduceLROnPlateau on train_loss instead
        if self.val_data_loader is None and isinstance(self.scheduler, ReduceLROnPlateau):
            # ReduceLROnPlateau scheduler takes metrics during its step call
            self.scheduler.step(metrics=train_loss)

        if self.config["trainer"]["use_tensorboard"]:
            self.tboard_writer.add_images('preprocessed image batch',
                                          next(iter(self.train_data_loader))[0],
                                          self.current_epoch)
            self.tboard_writer.add_scalar('Loss/train/epoch',
                                          train_loss,
                                          self.current_epoch)
            self.tboard_writer.add_scalar('Accuracy/train/epoch',
                                          train_accuracy,
                                          self.current_epoch)
            self.tboard_writer.add_scalar('Learning Rate/train/epoch',
                                          self.optimizer.param_groups[0]['lr'],
                                          self.current_epoch)

    def validate(self) -> None:
        """
        One cycle of model validation
        """
        t_0 = time.perf_counter()

        cumu_val_loss, y_true, _, y_pred = self.validate_one_epoch(
            self.val_data_loader)

        if self.data_set.val_set:
            val_size = len(self.data_set.val_set)
        else:  # take val set size from sampler if data_set.val_set is None
            val_size = len(self.val_data_loader.sampler.indices)
        val_loss = cumu_val_loss / val_size
        correct = sum(y_true == y_pred)
        val_accuracy = 100. * correct / val_size

        # add metrics to tracking best_metric_dicts
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
        self.logger.info('Validation set:')
        self.logger.info('\tVal time: %.2fs', (t_1 - t_0))
        self.logger.info('\tAverage loss: %.4f, Accuracy: %s/%s (%.0f%%)\n',
                         val_loss, correct, val_size, val_accuracy)
        if self.config["trainer"]["use_tensorboard"]:
            self.tboard_writer.add_scalar('Loss/validation/epoch',
                                          val_loss,
                                          self.current_epoch)
            self.tboard_writer.add_scalar('Accuracy/validation/epoch',
                                          val_accuracy,
                                          self.current_epoch)

    def test(self, weight_path: Optional[str] = None) -> None:
        """
        main test function
        args:
            weight_path: Path to pth weight file that will be loaded for test
                         Default is set to None which uses latest chkpt weight file
        """
        t_0 = time.perf_counter()
        if weight_path is not None:
            print(f"Loading new checkpoint from {weight_path} for testing")
            self.model = self.load_checkpoint(self.model, weight_path)
        if self.test_data_loader is None:
            raise NotImplementedError("test_data_loader is None or missing."
                                      "test_path might not have been set")
        cumu_test_loss, y_true, y_score, y_pred = self.validate_one_epoch(
            self.test_data_loader)
        t_1 = time.perf_counter()

        test_size = len(self.data_set.test_set)
        test_loss = cumu_test_loss / test_size

        self.logger.info('\nTest set:')
        self.logger.info('\tTest time: %.2fs', (t_1 - t_0))
        self.logger.info('\tAverage loss: %.4f', test_loss)
        if self.config["trainer"]["use_tensorboard"]:
            self.tboard_writer.add_scalar(
                'Loss/test/epoch',
                test_loss,
                self.current_epoch)
        num_classes = self.config["dataset"]["num_classes"]
        # add all test metrics
        for metric_name in self.config["metrics"]["test"]:
            metric_func = partial(getattr(module_metrics, metric_name))
            # set correct kwargs for metrics
            kwargs = {"y_true": y_true, "y_score": y_score, "y_pred": y_pred}
            if metric_name not in {"roc_auc_score"} and "y_score" in kwargs:
                del kwargs["y_score"]
            if metric_name in {"roc_auc_score"} and "y_pred" in kwargs:
                if num_classes > 1:
                    continue
                del kwargs["y_pred"]
            if num_classes > 1 and metric_name in {"f1_score", "precision_score", "recall_score"}:
                kwargs["average"] = "macro"

            metric_val = metric_func(**kwargs)
            self.logger.info('\t%s: %.4f', metric_name, metric_val)
            if self.config["trainer"]["use_tensorboard"]:
                self.tboard_writer.add_scalar(
                    f'{metric_name}/test/epoch',
                    metric_val,
                    self.current_epoch)

    def validate_one_epoch(self, data_loader: DataLoader) -> Tuple[float, List[int], List[float], List[int]]:
        """
        Evaluate model on one epoch with the given dataloader
        """
        self.model.eval()
        cumu_test_loss = 0
        y_true, y_score, y_pred = [], [], []
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model(data)
                # sum up batch loss
                cumu_test_loss += self.loss(output, target).item()
                # get the log-probability (proba) & index of the max log-probability (preds)
                proba, pred = F.softmax(output, dim=1).max(1)

                y_true.append(target)
                y_score.append(proba)
                y_pred.append(pred)
        y_true = torch.concat(y_true).cpu().numpy()
        y_score = torch.concat(y_score).cpu().numpy()
        y_pred = torch.concat(y_pred).cpu().numpy()

        return cumu_test_loss, y_true, y_score, y_pred

    def export_as_onnx(self, dummy_input: torch.Tensor, onnx_save_path: str = "checkpoints/onnx_model.onnx") -> None:
        """
        ONNX format export function
        Model should use torch tensors & torch operators for proper export
        numpy values are treated as constant nodes by the tracing method
        args:
            dummy_input: dummy input for the model
            onnx_save_path: path where onnx file will be saved
        """
        # Export with ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_save_path,
            export_params=True,
            do_constant_folding=True,
            opset_version=11,
            input_names=['input'],    # the model's input names
            output_names=['output'],  # the model's output names
            dynamic_axes={'input': {0: 'batch_size', 2: "height", 3: "width"},  # variable length axes
                          'output': {0: 'batch_size', 2: "height", 3: "width"}},
            verbose=False)

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
            image_list = [path for path in fpaths if osp.splitext(path)[-1] in IMG_EXTENSIONS]
        elif osp.isfile(source_path) and osp.splitext(source_path)[-1] in IMG_EXTENSIONS:
            image_list = [source_path]
        else:
            raise ValueError(
                f"Inference source {source_path} is not an image file or dir with images")

        pred_file_labels = []
        inference_transform = rgetattr(
            module_transforms, self.config["dataset"]["preprocess"]["inference_transform"])
        with torch.no_grad():
            for image_path in image_list:
                image = Image.open(image_path).convert('RGB')
                data = inference_transform(image)
                data = data.to(self.device).unsqueeze(0)
                output = self.model(data)
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1].item()

                self.logger.info("Image path=%s: predicted label=%s", image_path, pred)
                pred_file_labels.append([image_path, pred])
        if log_txt_preds:
            with open(osp.join(self.config.log_dir, "pred.txt"), 'w', encoding="utf-8") as pred_ptr:
                for image_path, pred in pred_file_labels:
                    pred_ptr.write(f"{image_path}, {pred}\n")

        self.logger.info("Inference complete for %s", source_path)
        return pred_file_labels

    def finalize_exit(self):
        """
        operations for graceful exit
        """
        print("Graceful exit initiating")
