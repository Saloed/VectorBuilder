import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class ConfMatrix:
    def __init__(self, classes, file_name):
        self.classes = classes
        self.file_name = file_name
        self._conf_mat = None
        self._data = None
        self._predicted = []
        self._actual = []

    def add(self, prediction, actual):
        self._actual.append(actual)
        self._predicted.append(prediction)

    def calc(self):
        """
        calculates normalized confusion matrix,
        precision, recall and f-score for accumulated data
        """
        cm = confusion_matrix(self._actual, self._predicted)
        self._conf_mat = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]
        p, r, f, _ = precision_recall_fscore_support(self._actual, self._predicted)
        self._data = (p, r, f, [np.mean(f)])
        return p, r, f

    def __str__(self):
        if self._conf_mat is not None and self._data is not None:
            labels = ['Precision', 'Recall\t', 'F-Score\t', 'Avg F-Score\t']
            return '\n'.join([
                'Confusion Matrix',
                str(self._conf_mat),
                *('\t'.join([label, *('{0:.3f} & '.format(v) for i, v in enumerate(self._data[j]))])
                  for j, label in enumerate(labels))
            ])
        else:
            return 'Confusion matrix is not calculated yet.'

    def print_conf_matrix(self):
        cm = self._conf_mat
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(self.classes)
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, '{0:.3f}'.format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Actual class')
        plt.xlabel('Predicted class')
        plt.savefig(self.file_name + 'conf_matrix.png')
        plt.show()
