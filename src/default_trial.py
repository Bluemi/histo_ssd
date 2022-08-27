from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torchvision
import torch.nn.functional as functional
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from torchvision import transforms
from torchmetrics.detection import MeanAveragePrecision
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, LRScheduler, TorchData, DataLoader
from determined.tensorboard.metric_writers.pytorch import TorchWriter
import matplotlib.pyplot as plt

from datasets import LizardDetectionDataset
from datasets.augmentation_wrapper import AugmentationWrapper
from datasets.banana_dataset import BananasDataset
from models import SSDModel, predict
from utils.bounding_boxes import multibox_target
from utils.funcs import draw_boxes
from utils.metrics import update_mean_average_precision, calc_loss, calc_cls_bbox_loss

DEFAULT_WARMUP_BATCHES = 300


class DefaultTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        super().__init__(context)
        self.context = context

        # the dataset is loaded at the start to make it possible to split it
        self.train_dataset, self.validation_dataset = self._load_dataset()
        self.num_classes = self._get_num_classes()
        self.negative_ratio = self.context.get_hparams().get('negative_ratio')
        self.normalize_per_batch = self.context.get_hparams().get('hnm_norm_per_batch', True)
        assert isinstance(self.normalize_per_batch, bool)

        if self.negative_ratio is None:
            print('WARN: hard negative mining is disabled')

        # create model
        backbone_arch = self.context.get_hparam('backbone_arch')
        smin = self.context.get_hparams().get('min_anchor_size', 0.2)
        smax = self.context.get_hparams().get('max_anchor_size', 0.9)
        self.pretrained = self.context.get_hparam('pretrained')
        self.warmup_batches = self.context.get_hparams().get('warmup_batches', DEFAULT_WARMUP_BATCHES)

        if self.pretrained:
            model = SSDModel.from_state_dict(
                state_dict_path='DOWNLOAD', num_classes=self.num_classes, backbone_arch=backbone_arch,
                min_anchor_size=smin, max_anchor_size=smax, freeze_pretrained=True
            )
        else:
            model = SSDModel(
                num_classes=self.num_classes, backbone_arch=backbone_arch, min_anchor_size=smin, max_anchor_size=smax
            )

        # pred layer
        self.model = self.context.wrap_model(model)

        optimizer_name = self.context.get_hparam('optimizer')
        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.context.get_hparam('learning_rate'),
                momentum=self.context.get_hparams().get('momentum', 0),
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

        self.tblogger = TorchWriter()  # Tensorboard log

    def _load_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        Loads the dataset and splits it into train and validation.

        :return: Tuple with (train_dataset, validation_dataset)
        """
        dataset_name = self.context.get_hparam('dataset')
        split_size = self.context.get_hparam('dataset_split_size')
        image_size = self.context.get_hparams().get('image_size', 224)
        force_one_class = self.context.get_hparams().get('force_one_class', False)

        print('loading \"{}\" dataset: '.format(dataset_name), end='', flush=True)
        if dataset_name == 'lizard':
            dataset = LizardDetectionDataset.from_avocado(
                image_size=np.array([image_size, image_size]),
                image_stride=np.array([image_size, image_size]),
                use_cache=True,
                show_progress=False,
                force_one_class=force_one_class,
            )
            datasets = dataset.split(split_size)
        elif dataset_name == 'banana':
            dataset_location = '/data/ldap/histopathologic/original_read_only/banana-detection'
            dataset_train = BananasDataset(
                data_dir=dataset_location, is_train=True, verbose=False,
                transforms=[torchvision.transforms.Resize((image_size, image_size))]
            )
            dataset_val = BananasDataset(
                data_dir=dataset_location, is_train=False, verbose=False,
                transforms=[torchvision.transforms.Resize((image_size, image_size))]
            )
            datasets = (dataset_train, dataset_val)
        else:
            raise ValueError('Unknown dataset: {}'.format(dataset_name))

        print('Done', flush=True)

        return datasets

    def _get_num_classes(self) -> int:
        """
        Defines the number of classes for the given dataset

        :return: The number of classes
        """
        dataset_name = self.context.get_hparam('dataset')
        if dataset_name == 'lizard':
            force_one_class = self.context.get_hparams().get('force_one_class', False)
            if force_one_class:
                return 1
            else:
                return 6
        elif dataset_name == 'banana':
            return 1
        else:
            raise ValueError('Unknown dataset: {}'.format(dataset_name))

    @staticmethod
    def _get_max_class_probs(cls_preds: torch.Tensor):
        cls_probs = functional.softmax(cls_preds, dim=2)
        cls_probs = cls_probs.reshape((-1, cls_probs.shape[-1]))
        return torch.max(cls_probs, dim=0)[0]

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, Any]:
        image = batch['image']
        boxes = batch['boxes']
        self.optimizer.zero_grad()

        if self.pretrained and batch_idx >= self.warmup_batches:
            self.model.unfreeze()
            self.pretrained = False  # dont unfreeze again

        anchors, cls_preds, bbox_preds = self.model(image)
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, boxes)
        cls_loss, bbox_loss = calc_cls_bbox_loss(
            cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, negative_ratio=self.negative_ratio,
            normalize_per_batch=self.normalize_per_batch
        )
        loss = (cls_loss + bbox_loss).mean()
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        result = {
            'loss': loss,
            'bbox_loss': bbox_loss.mean(),
            'cls_loss': cls_loss.mean(),
            'scheduler_lr': self.scheduler.get_last_lr()[0],
        }

        if loss.isnan().any():
            raise ValueError('Got NaN loss')

        class_max_probs = DefaultTrial._get_max_class_probs(cls_preds)
        for i, cls_max_prob in enumerate(class_max_probs):
            result['cls{}_max_prob'.format(i)] = cls_max_prob

        return result

    def write_prediction_images(
            self, batch_output: List[torch.Tensor], batch: Dict[str, torch.Tensor], batch_idx: int, image_counter: int
    ) -> int:
        """
        Writes prediction images to tblogger.

        :param batch_output: The output of the model
        :param batch: The ground truth batch
        :param batch_idx: The current batch index
        :param image_counter: The number of images already logged for this epoch.
        """
        image_prediction_max_images = self.context.get_hparam('image_prediction_max_images')
        for image, boxes, output in zip(batch['image'], batch['boxes'], batch_output):
            image = image.to(self.context.device)
            draw_image = (image * 255.0).squeeze(0).permute(1, 2, 0).long()

            # draw predictions
            for row in output:
                score = float(row[1])
                if score < self.context.get_hparam('image_prediction_score_threshold'):
                    continue
                bbox = row[2:6].unsqueeze(0)
                draw_boxes(draw_image, bbox, color=(255, 0, 0), box_format='ltrb')
            # draw ground truth
            for box in boxes:
                if box[0] < 0:
                    continue
                bbox = box[1:5].unsqueeze(0)
                draw_boxes(draw_image, bbox, color=(0, 255, 0), box_format='ltrb')

            fig = plt.figure(figsize=(10, 10))
            plt.imshow(draw_image.cpu())
            self.tblogger.writer.add_figure(
                f'Prediction_batch_{batch_idx}_{image_counter}',
                fig,
                global_step=batch_idx
            )
            image_counter += 1
            if image_counter >= image_prediction_max_images:
                return image_counter

    def evaluate_full_dataset(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Calculates the mean average precision for the full evaluation dataset.
        Also shows some images of predictions of the model

        :param data_loader: The dataloader of the evaluation dataset.
        :return: Dict containing mAP value
        """
        # noinspection PyProtectedMember
        batch_idx = self.context._current_batch_idx + 1
        image_prediction_max_images = self.context.get_hparam('image_prediction_max_images')

        mean_average_precision = MeanAveragePrecision(box_format='xyxy', class_metrics=True)

        image_counter = 0
        losses = []
        all_max_class_probs = []
        cls_loss = None
        bbox_loss = None
        for batch in data_loader:
            anchors, cls_preds, bbox_preds = self.model(batch['image'].to(self.context.device))

            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, batch['boxes'].to(self.context.device))
            cls_loss, bbox_loss = calc_cls_bbox_loss(
                cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, negative_ratio=self.negative_ratio
            )
            loss = (cls_loss + bbox_loss).mean()
            losses.append(loss)

            batch_output = predict(anchors, cls_preds, bbox_preds)

            update_mean_average_precision(mean_average_precision, batch['boxes'], batch_output)

            max_class_probs = DefaultTrial._get_max_class_probs(cls_preds)
            all_max_class_probs.append(max_class_probs)

            # write prediction images
            if image_counter < image_prediction_max_images:
                image_counter = self.write_prediction_images(batch_output, batch, batch_idx, image_counter)

        # TODO: result['map_per_class'] should be returned separate for each class
        result = mean_average_precision.compute()

        result['loss'] = torch.mean(torch.tensor(losses)).item()
        result['cls_loss'] = torch.mean(cls_loss)
        result['bbox_loss'] = torch.mean(bbox_loss)

        class_max_probs = torch.max(torch.stack(all_max_class_probs), dim=0)[0]
        for i, cls_max_prob in enumerate(class_max_probs):
            result['cls{}_max_prob'.format(i)] = cls_max_prob

        return result

    def build_training_data_loader(self) -> DataLoader:
        # augmentation
        augmentation_stack = self.context.get_hparams().get('augmentation_stack')
        if augmentation_stack is None:
            dataset = self.train_dataset
        elif augmentation_stack == 'minimal':
            # noinspection PyUnresolvedReferences
            mean, std = self.train_dataset.normalization_values()
            dataset = AugmentationWrapper(self.train_dataset, [transforms.Normalize(mean, std)])
        else:
            raise ValueError('Unknown augmentation stack: {}'.format(augmentation_stack))

        return DataLoader(
            dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=self.context.get_hparam('num_workers'),
            pin_memory=True,
        )

    def build_validation_data_loader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False,  # set to True, to see different images in Tensorboard
            num_workers=self.context.get_hparam('num_workers'),
            pin_memory=True
        )
