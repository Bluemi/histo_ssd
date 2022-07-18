from typing import Tuple, Dict, Any

import numpy as np
import torch
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, LRScheduler, TorchData, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset

from datasets import LizardDetectionDataset
from models import TinySSD
from utils.bounding_boxes import multibox_target


class DefaultTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        super().__init__(context)
        self.context = context

        # the dataset is loaded at the start to make it possible to split it
        self.train_dataset, self.validation_dataset = self._load_dataset()

        # Creates a feature vector
        model = self.context.get_hparam('model')
        if model == 'tiny_ssd':
            network = TinySSD(num_classes=6)
        else:
            raise ValueError('Unknown model \"{}\"'.format(model))

        # pred layer
        self.network = self.context.wrap_model(network)

        optimizer_name = self.context.get_hparam('optimizer')
        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.context.get_hparam('learning_rate'),
                # momentum=0.9, TODO: try momentum
                weight_decay=self.context.get_hparam('l2_regularization')
            )
        else:
            raise ValueError('Could not find optimizer "{}"'.format(optimizer_name))
        self.optimizer = self.context.wrap_optimizer(optimizer)

        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.context.get_experiment_config()['searcher']['max_length']['batches']
        )
        self.scheduler = self.context.wrap_lr_scheduler(scheduler, LRScheduler.StepMode.STEP_EVERY_BATCH)

        self.cls_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.bbox_loss = torch.nn.L1Loss(reduction='none')

    def _load_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        Loads the dataset and splits it into train and validation.

        :return: Tuple with (train_dataset, validation_dataset)
        """
        dataset = self.context.get_hparam('dataset')
        split_size = self.context.get_hparam('dataset_split_size')
        if dataset == 'lizard':
            dataset = LizardDetectionDataset.from_avocado(
                image_size=np.array([224, 224]),
                image_stride=np.array([224, 224]),
                use_cache=True,
                show_progress=False,
            )
            return dataset.split(split_size)

    def _calc_loss(self, class_preds, class_labels, bounding_box_preds, bounding_box_labels, bounding_box_masks):
        batch_size, num_classes = class_preds.shape[0], class_preds.shape[2]
        cls = self.cls_loss(
            class_preds.reshape(-1, num_classes), class_labels.reshape(-1)
        ).reshape(batch_size, -1).mean(dim=1)
        bbox = self.bbox_loss(
            bounding_box_preds * bounding_box_masks, bounding_box_labels * bounding_box_masks
        ).mean(dim=1)
        return cls + bbox

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, Any]:
        image = batch['image']
        boxes = batch['boxes']
        self.optimizer.zero_grad()

        anchors, cls_preds, bbox_preds = self.network(image)
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, boxes)
        loss = self._calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks).mean()
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {
            'loss': loss,
        }

    def evaluate_batch(self, batch: TorchData, batch_idx: int) -> Dict[str, Any]:
        image = batch['image']
        boxes = batch['boxes']

        with torch.no_grad():
            anchors, cls_preds, bbox_preds = self.network(image)
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, boxes)
            loss = self._calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks).mean()

        return {
            'loss': loss,
        }

    def build_training_data_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=self.context.get_hparam('num_workers'),
            pin_memory=True
        )

    def build_validation_data_loader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False,
            num_workers=self.context.get_hparam('num_workers'),
            pin_memory=True
        )

