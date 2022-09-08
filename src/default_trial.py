from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from torchmetrics.detection import MeanAveragePrecision
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, LRScheduler, TorchData, DataLoader
from determined.tensorboard.metric_writers.pytorch import TorchWriter
import matplotlib.pyplot as plt

from datasets import LizardDetectionDataset
from datasets.augmentation_wrapper import AugmentationWrapper
from datasets.banana_dataset import BananasDataset
from models import SSDModel, predict
from utils.bounding_boxes import multibox_target
from utils.clock import Clock
from utils.funcs import draw_boxes, DARK_COLORS, BRIGHT_COLORS
from utils.metrics import update_mean_average_precision, calc_cls_bbox_loss, update_confusion_matrix, ConfusionMatrix
from utils.augmentations import RandomRotate, RandomFlip

WRITE_PREDICTIONS_BATCH = 2900
NUM_PRED_LIMIT = 700  # limit number of predictions per sample (there are samples with 666 ground truth boxes)
# only use some samples for mean average precision update. Only allow predictions for 600 images
MAX_MAP_UPDATES = NUM_PRED_LIMIT * 100
USE_MAP_UNDIV = False


class DefaultTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        super().__init__(context)
        self.context = context

        # set hyperparameters
        self.negative_ratio = self.context.get_hparams().get('negative_ratio')
        if self.negative_ratio is None:
            print('WARN: hard negative mining is disabled')
        self.normalize_per: str = self.context.get_hparams().get('hnm_norm_per', 'none')
        assert isinstance(self.normalize_per, str)
        self.use_smooth_l1 = self.context.get_hparams().get('use_smooth_l1', True)
        self.nms_threshold = self.context.get_hparams().get('nms_threshold', 0.5)
        backbone_arch = self.context.get_hparam('backbone_arch')
        smin = self.context.get_hparams().get('min_anchor_size', 0.2)
        smax = self.context.get_hparams().get('max_anchor_size', 0.9)
        self.pretrained = self.context.get_hparams().get('pretrained', False)
        self.warmup_batches = self.context.get_hparams().get('warmup_batches')
        self.enable_class_metrics = self.context.get_hparams().get('enable_class_metrics', False)
        self.use_clock = self.context.get_hparams().get('use_clock', False)
        ignore_classes = self.context.get_hparams().get('ignore_classes')
        if ignore_classes is not None:
            ignore_classes = list(map(int, ignore_classes.split(',')))
        self.ignore_classes = ignore_classes
        self.num_classes = self._get_num_classes()
        optimizer_name = self.context.get_hparam('optimizer')
        self.max_eval_time = self.context.get_hparams().get('max_eval_time')
        self.bbox_loss_scale = self.context.get_hparams().get('bbox_loss_scale', 1.0)
        self.dataset_image_size = self.context.get_hparam('dataset_image_size')
        self.always_compute_map = self.context.get_hparams().get('always_compute_map', False)
        self.iou_match_threshold = self.context.get_hparams().get('iou_match_threshold', 0.5)
        self.use_center_points = self.context.get_hparams().get('use_center_points', False)

        # the dataset is loaded at the start to make it possible to split it
        self.train_dataset, self.validation_dataset = self._load_dataset()
        self.enable_full_evaluation = False

        # create model
        if self.pretrained:
            model = SSDModel.from_state_dict(
                state_dict_path='DOWNLOAD', num_classes=self.num_classes, backbone_arch=backbone_arch,
                min_anchor_size=smin, max_anchor_size=smax, freeze_pretrained=False,
                center_points=self.use_center_points,
            )
        else:
            model = SSDModel(
                num_classes=self.num_classes, backbone_arch=backbone_arch, min_anchor_size=smin, max_anchor_size=smax,
                center_points=self.use_center_points,
            )

        # noinspection PyTypeChecker
        self.model: SSDModel = self.context.wrap_model(model)

        if self.pretrained:
            self.model.freeze_backbone()

        # optimizer
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

        # scheduler
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.context.get_experiment_config()['searcher']['max_length']['batches']
        )
        self.scheduler = self.context.wrap_lr_scheduler(scheduler, LRScheduler.StepMode.STEP_EVERY_BATCH)

        # tensorboard logger
        self.tblogger = TorchWriter()

    def _load_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        Loads the dataset and splits it into train and validation.

        :return: Tuple with (train_dataset, validation_dataset)
        """
        dataset_name = self.context.get_hparam('dataset')
        image_stride = self.context.get_hparams().get('image_stride', self.dataset_image_size)
        if isinstance(image_stride, float):
            image_stride = round(self.dataset_image_size * image_stride)
        force_one_class = self.context.get_hparams().get('force_one_class', False)

        print('loading \"{}\" dataset...'.format(dataset_name))
        if dataset_name == 'lizard':
            dataset = LizardDetectionDataset.from_avocado(
                image_size=np.array([self.dataset_image_size, self.dataset_image_size]),
                image_stride=np.array([image_stride, image_stride]),
                use_cache=True,
                show_progress=False,
                force_one_class=force_one_class,
                ignore_classes=self.ignore_classes,
            )
            split_size = self.context.get_hparam('dataset_split_size')
            datasets = dataset.split(split_size)
            print('train len: {}  val len: {}'.format(len(datasets[0]), len(datasets[1])))
        elif dataset_name == 'banana':
            dataset_location = '/data/ldap/histopathologic/original_read_only/banana-detection'
            dataset_train = BananasDataset(
                data_dir=dataset_location, is_train=True, verbose=False,
            )
            dataset_val = BananasDataset(
                data_dir=dataset_location, is_train=False, verbose=False,
            )
            datasets = (dataset_train, dataset_val)
        else:
            raise ValueError('Unknown dataset: {}'.format(dataset_name))

        print('Done')

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
                num_removed_classes = 0 if self.ignore_classes is None else len(self.ignore_classes)
                return 6 - num_removed_classes
        elif dataset_name == 'banana':
            return 1
        else:
            raise ValueError('Unknown dataset: {}'.format(dataset_name))

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, Any]:
        image = batch['image']
        boxes = batch['boxes']
        self.optimizer.zero_grad()

        if batch_idx > WRITE_PREDICTIONS_BATCH:
            self.enable_full_evaluation = True

        if self.pretrained and batch_idx >= self.warmup_batches:
            self.model.unfreeze()
            self.pretrained = False  # dont unfreeze again

        anchors, cls_preds, bbox_preds = self.model(image)
        bbox_labels, bbox_masks, cls_labels = multibox_target(
            anchors, boxes, self.use_center_points, iou_match_threshold=self.iou_match_threshold
        )
        cls_loss, bbox_loss = calc_cls_bbox_loss(
            cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, negative_ratio=self.negative_ratio,
            normalize_per=self.normalize_per, use_smooth_l1=self.use_smooth_l1,
        )
        loss = (cls_loss + bbox_loss * self.bbox_loss_scale).mean()
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
        image_prediction_threshold = self.context.get_hparam('image_prediction_score_threshold')
        for image, ground_truth_boxes, output in zip(batch['image'], batch['boxes'], batch_output):
            image = image.to(self.context.device)
            draw_image = (image * 255.0).squeeze(0).permute(1, 2, 0).long()

            # draw ground truth
            draw_boxes(
                draw_image, ground_truth_boxes[:, 1:],  box_format='ltrb',
                color=DARK_COLORS, color_indices=ground_truth_boxes[:, 0],
            )
            # draw predictions
            prediction_sign = 'cross' if self.use_center_points else 'box'
            shown_output = output[output[:, 1] > image_prediction_threshold]
            draw_boxes(
                draw_image, shown_output[:, 2:], box_format='ltrb',
                color=BRIGHT_COLORS, color_indices=shown_output[:, 0], sign=prediction_sign,
            )

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
        eval_clock = Clock()
        # noinspection PyProtectedMember
        batch_idx = self.context._current_batch_idx + 1
        image_prediction_max_images = self.context.get_hparam('image_prediction_max_images')

        mean_ap = MeanAveragePrecision(
            box_format='xyxy', class_metrics=self.enable_class_metrics,
            # max_detection_thresholds=[600, 600, 600],  # sometimes we have 600 predictions for one image
        )

        mean_ap_undiv = MeanAveragePrecision(
            box_format='xyxy', class_metrics=self.enable_class_metrics,
            # max_detection_thresholds=[600, 600, 600],  # sometimes we have 600 predictions for one image
        )

        confusion_matrix = ConfusionMatrix()

        calculate_map = self.enable_full_evaluation or self.always_compute_map

        image_counter = 0
        mean_average_precision_counter = 0
        losses = []
        cls_loss = None
        bbox_loss = None
        go_through_dataset_clock = Clock()
        for batch in data_loader:
            anchors, cls_preds, bbox_preds = self.model(batch['image'].to(self.context.device))

            bbox_labels, bbox_masks, cls_labels = multibox_target(
                anchors, batch['boxes'].to(self.context.device), self.use_center_points,
                iou_match_threshold=self.iou_match_threshold
            )
            # don't use negative_ratio-hparam or norm_per_batch-hparam for evaluation
            cls_loss, bbox_loss = calc_cls_bbox_loss(
                cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, negative_ratio=3.0,
                use_smooth_l1=self.use_smooth_l1
            )
            loss = (cls_loss + bbox_loss).mean()  # we do not scale loss here for evaluation
            losses.append(loss)

            predict_clock = Clock()
            pos_threshold = 0.5
            if self.enable_full_evaluation:
                pos_threshold = 0.2
            batch_output = predict(
                anchors, cls_preds, bbox_preds, nms_iou_threshold=self.nms_threshold, pos_threshold=pos_threshold,
                num_pred_limit=NUM_PRED_LIMIT,
            )

            # check num outputs
            for out in batch_output:
                if len(out) == NUM_PRED_LIMIT:
                    print('WARN: limited output to {} predictions'.format(len(out)))
                    break

            last_predict_duration = predict_clock.stop()

            # mean average precision
            if calculate_map:
                if self.enable_full_evaluation or mean_average_precision_counter < MAX_MAP_UPDATES:
                    update_mean_average_precision(mean_ap, batch['boxes'], batch_output, divide_limit=100)
                    if USE_MAP_UNDIV:
                        update_mean_average_precision(mean_ap_undiv, batch['boxes'], batch_output)
                    for out in batch_output:
                        mean_average_precision_counter += len(out)
                    if not (self.enable_full_evaluation or mean_average_precision_counter < MAX_MAP_UPDATES):
                        print('WARN: stopping map updates')

            # confusion matrix
            update_confusion_matrix(confusion_matrix, batch['boxes'], batch_output)

            # write prediction images
            if self.enable_full_evaluation and image_counter < image_prediction_max_images:
                image_counter = self.write_prediction_images(batch_output, batch, batch_idx, image_counter)
            if self.max_eval_time is not None and eval_clock.get_duration() > self.max_eval_time:
                print('early evaluation stop. Took {} sec until now. Last predict() took {} sec'.format(
                    eval_clock.get_duration(), last_predict_duration
                ))
                break  # stop early, if it takes too long
        if self.use_clock:
            go_through_dataset_clock.sap('predict dataset')

        # TODO: result['map_per_class'] should be returned separate for each class
        map_clock = Clock()
        result = {}
        if calculate_map:
            result = mean_ap.compute()

        if self.use_clock:
            map_clock.sap('map.compute() for {} samples'.format(mean_average_precision_counter))
        if calculate_map and USE_MAP_UNDIV:
            map_clock.start()
            result_undiv = mean_ap_undiv.compute()
            if self.use_clock:
                map_clock.sap('map_undiv.compute() for {} samples'.format(mean_average_precision_counter))

            for key, value in result_undiv.items():
                result['{}_undiv'.format(key)] = value

        result['loss'] = torch.mean(torch.tensor(losses)).item()
        result['cls_loss'] = torch.mean(cls_loss)
        result['bbox_loss'] = torch.mean(bbox_loss)
        result['precision'] = confusion_matrix.precision()
        result['recall'] = confusion_matrix.recall()
        result['f1_score'] = confusion_matrix.f1_score()

        if self.use_clock:
            eval_clock.sap('eval dataset')

        return result

    def build_training_data_loader(self) -> DataLoader:
        # augmentation
        transforms = []

        use_normalization = self.context.get_hparams().get('aug_norm', False)
        if use_normalization:
            # noinspection PyUnresolvedReferences
            mean, std = self.train_dataset.normalization_values()
            transforms.append(
                ('image', torchvision.transforms.Normalize(mean, std))
            )

        use_rotate = self.context.get_hparams().get('aug_rotate', False)
        if use_rotate:
            transforms.append(
                (None, RandomRotate())
            )

        use_flip = self.context.get_hparams().get('aug_flip', False)
        if use_flip:
            transforms.append(
                (None, RandomFlip())
            )

        model_image_size = self.context.get_hparam('model_image_size')
        if model_image_size != self.dataset_image_size:
            transforms.append(
                ('image', torchvision.transforms.Resize((model_image_size, model_image_size)))
            )

        dataset = AugmentationWrapper(
            self.train_dataset,
            transforms,
        )

        return DataLoader(
            dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=self.context.get_hparam('num_workers'),
            pin_memory=True,
        )

    def build_validation_data_loader(self) -> DataLoader:
        shuffle_validation = self.context.get_hparams().get('shuffle_validation', False)

        transforms = []

        model_image_size = self.context.get_hparam('model_image_size')
        if model_image_size != self.dataset_image_size:
            transforms.append(
                ('image', torchvision.transforms.Resize((model_image_size, model_image_size)))
            )

        dataset = AugmentationWrapper(
            self.validation_dataset,
            transforms
        )

        return DataLoader(
            dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=shuffle_validation,  # set to True, to see different images in Tensorboard
            num_workers=self.context.get_hparam('num_workers'),
            pin_memory=True
        )
