import torch
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.functional import average_precision, precision_recall_curve, auc
from torchmetrics import BinnedAveragePrecision
import matplotlib.pyplot as plt
from sklearn import metrics

from utils.bounding_boxes import intersection_over_union


class MAP:
    def __init__(self, num_classes, iou_thresholds=None):
        self.num_classes = num_classes
        if iou_thresholds is None:
            iou_thresholds = torch.linspace(0.5, 0.95, 10)
        self.iou_thresholds = iou_thresholds

        self.pred_boxes = []  # every entry corresponds to an image
        self.pred_labels = []
        self.pred_scores = []
        self.target_boxes = []
        self.target_labels = []

    def update(self, preds, target):
        """

        :param preds: A list of dicts with keys 'boxes', 'labels', 'scores'
        :param target: A list of dicts with keys 'boxes', 'labels'
        :return:
        """
        for pred, target in zip(preds, target):
            self.pred_boxes.append(pred['boxes'])
            self.pred_labels.append(pred['labels'])
            self.pred_scores.append(pred['scores'])
            self.target_boxes.append(target['boxes'])
            self.target_labels.append(target['labels'])

    def compute(self):
        all_ious = []
        map_75 = None
        map_50 = None
        # for iou_threshold in self.iou_thresholds:
        for iou_threshold in (0.5,):
            all_classes = []
            for class_id in range(self.num_classes):
                all_samples = []
                for sample_id in range(len(self.pred_boxes)):
                    mean_ap = self._compute(iou_threshold, class_id, sample_id)
                    all_samples.append(mean_ap)
                all_classes.append(torch.mean(torch.tensor(all_samples)))
            mean_all_classes = torch.mean(torch.tensor(all_classes))
            all_ious.append(mean_all_classes)
            if iou_threshold == 0.5:
                map_50 = mean_all_classes
            elif iou_threshold == 0.75:
                map_75 = mean_all_classes
        all_mean = torch.mean(torch.tensor(all_ious))
        return {
            # 'map': all_mean,
            # 'map_75': map_75,
            'map_50': map_50,
        }

    def _compute(self, iou_threshold, class_id, sample_id):
        # get relevant preds
        pred_indices = self.pred_labels[sample_id] == class_id
        pred_boxes = self.pred_boxes[sample_id][pred_indices]
        pred_scores = self.pred_scores[sample_id][pred_indices]

        # get relevant targets
        target_indices = self.target_labels[sample_id] == class_id
        target_boxes = self.target_boxes[sample_id][target_indices]

        ap_preds = []
        ap_targets = []
        # calc iou TODO: cache
        iou = intersection_over_union(pred_boxes, target_boxes)
        # n_matches = 0
        for pred_index in torch.argsort(pred_scores):
            max_iou, target_index = torch.max(iou[pred_index], dim=0)
            score = pred_scores[pred_index]
            if max_iou > iou_threshold:
                ap_preds.append(score)
                ap_targets.append(1)
                iou[pred_index, :] = -1
                iou[:, target_index] = -1
            else:
                ap_preds.append(score)
                ap_targets.append(0)

        # for _ in range(len(target_boxes) - n_matches):
        #     ap_preds.append(1)
        #     ap_targets.append(0)

        # print(ap_preds, ap_targets)
        ap = average_precision(torch.tensor(ap_preds), torch.tensor(ap_targets))
        if ap.isnan():
            ap = torch.tensor(0.)
        return ap


def main():
    preds = [
        dict(
            boxes=torch.tensor([
                [0.6, 0.6, 0.7, 0.7],  # miss
                [0.2, 0.2, 0.4, 0.4],  # hit
            ]),
            scores=torch.tensor([
                0.3,
                0.5
            ]),
            labels=torch.tensor([0, 0]),
        )
    ]
    target = [
        dict(
            boxes=torch.tensor([
                [0.5, 0.5, 0.6, 0.6],  # miss
                [0.2, 0.2, 0.4, 0.4],  # hit
            ]),
            labels=torch.tensor([0, 0]),
        )
    ]
    for name, mean_ap in [('torch', MeanAveragePrecision()), ('my', MAP(1))]:
        print('{}:'.format(name))
        mean_ap.update(preds, target)
        result = mean_ap.compute()
        # for key in ('map', 'map_50', 'map_75'):
        for key in ('map_50',):
            print('  {}: {}'.format(key, result[key]))
    print('\n')


def test():
    binned_ap = BinnedAveragePrecision(num_classes=1, thresholds=101)
    ap_preds = torch.tensor([0.0, 0.3, 0.5])
    ap_targets = torch.tensor([1, 0, 1])

    ap_sklearn = metrics.average_precision_score(ap_targets, ap_preds)
    ap = average_precision(ap_preds, ap_targets)
    binned_ap.update(ap_preds, ap_targets)
    bap = binned_ap.compute()
    precision, recall, thresholds = precision_recall_curve(ap_preds, ap_targets)
    auc_val = auc(recall, precision)

    plt.plot(recall, precision)
    plt.xlabel('{:.3f}'.format(ap))
    print('ap sklearn: {:.3f}'.format(ap_sklearn))
    print('ap        : {:.3f}'.format(ap))
    print('bap       : {:.3f}'.format(bap))
    print('auc       : {:.3f}'.format(auc_val))
    threshold_list = ', '.join(list(map('{:.2f}'.format, thresholds.tolist())))
    print('thresholds:', threshold_list)
    plt.show()


if __name__ == '__main__':
    main()
    test()
