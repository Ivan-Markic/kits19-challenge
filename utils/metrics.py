import numpy as np


class Evaluator:
    def __init__(self, num_classes):
        np.seterr(divide='ignore', invalid='ignore')
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.dc_per_case = np.zeros(self.num_classes)
        self.acc_per_case = np.zeros(self.num_classes)
        self.iou_per_case = np.zeros(self.num_classes)

        self.dc_each_case = []
        self.acc_each_case = []
        self.iou_each_case = []
        self.num_case = np.zeros(self.num_classes)

    def _generate_matrix(self, pred, label):
        mask = (label >= 0) & (label < self.num_classes)
        label = self.num_classes * label[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    # Calculating metrics
    def _calculate_metrics(self, class_, confusion_matrix):
        intersection = confusion_matrix[class_][class_] + 1e-10

        if np.isnan(intersection):
            intersection = -1

        overlapped_union = np.sum(confusion_matrix, axis=0)[class_] + np.sum(confusion_matrix, axis=1)[class_] + 1e-10

        if np.isnan(overlapped_union):
            overlapped_union = -1

        intersection_over_union = intersection / (overlapped_union - intersection)

        if np.isnan(intersection_over_union):
            intersection_over_union = -1

        # Dice coefficient calculation
        dc = intersection * 2 / overlapped_union
        if np.isnan(dc):
            dc = -1

        # Basic Accuracy calculation
        total_pixels = overlapped_union - intersection
        accuracy = intersection / total_pixels if total_pixels > 0 else 0

        return dc, accuracy, intersection_over_union

    def add_batch(self, preds, labels):
        assert preds.shape == labels.shape
        for i in range(len(preds)):
            self.add(preds[i], labels[i])

    def add(self, pred, label):
        assert pred.shape == label.shape
        matrix = self._generate_matrix(pred, label)
        self.confusion_matrix += matrix

        dc_case = np.zeros(self.num_classes)
        acc_case = np.zeros(self.num_classes)
        iou_case = np.zeros(self.num_classes)
        for cls in range(self.num_classes):
            dc, acc, iou = self._calculate_metrics(cls, matrix)
            if dc != -1:
                self.dc_per_case[cls] += dc
            if acc != -1:
                self.acc_per_case[cls] += acc
            if iou != -1:
                self.iou_per_case[cls] += iou
            self.num_case[cls] += 1
            dc_case[cls] = dc
            acc_case[cls] = acc
            iou_case[cls] = iou

        self.dc_each_case.append(dc_case)
        self.acc_each_case.append(acc_case)
        self.iou_each_case.append(iou_case)

    def eval(self):
        acc = dict()
        for cls in range(self.num_classes):
            dc_per_case = self.dc_per_case[cls] / self.num_case[cls]
            acc_per_case = self.acc_per_case[cls] / self.num_case[cls]
            iou_per_case = self.iou_per_case[cls] / self.num_case[cls]
            dc_global, acc_global, iou_global = self._calculate_metrics(cls, self.confusion_matrix)
            # per case
            acc[f'dc_per_case_{cls}'] = dc_per_case
            acc[f'acc_per_case_{cls}'] = acc_per_case
            acc[f'iou_per_case_{cls}'] = iou_per_case

            # global
            acc[f'dc_global_{cls}'] = dc_global
            acc[f'acc_global_{cls}'] = acc_global
            acc[f'iou_global_{cls}'] = iou_global
        acc[f'dc_each_case'] = self.dc_each_case
        acc[f'acc_each_case'] = self.acc_each_case
        acc[f'iou_each_case'] = self.iou_each_case
        return acc
