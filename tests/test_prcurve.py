import numpy as np
from sklearn.metrics import precision_recall_curve

from matplotlib import pyplot as plt


def main():
    y_true      = ["p", "n", "n", "p", "p" , "p", "n", "p", "n", "p", "p", "p", "p", "n", "n", "n"]
    pred_scores = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    # pred_scores = [0.7, 0.3, 0.5, 0.6, 0.55, 0.9, 0.4, 0.2, 0.4, 0.3, 0.7, 0.5, 0.8, 0.2, 0.3, 0.35]
    # thresholds = np.arange(start=0.2, stop=0.7, step=0.05)
    precisions, recalls, thresholds = precision_recall_curve(
        y_true=y_true, probas_pred=pred_scores, pos_label='p'
    )
    plt.plot(recalls, precisions, linewidth=4, color="blue")
    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    plt.show()


if __name__ == '__main__':
    main()
