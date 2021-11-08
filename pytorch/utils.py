from __future__ import absolute_import, division, print_function

import os
import logging
import torch
import pandas as pd


class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args


class Logger:
    def __init__(self, info=logging.INFO, name=__name__):
        logger = logging.getLogger(name)
        logger.setLevel(info)

        self.__logger = logger

    def get_logger(self, handler_type='stream_handler'):
        if handler_type == 'stream_handler':
            handler = logging.StreamHandler()
            log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(log_format)
        else:
            handler = logging.FileHandler('utils.log')

        self.__logger.addHandler(handler)

        return self.__logger


def confusion_matrix(predictions, test_labels, class_name_list, path):
    num_classes = len(class_name_list)
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(predictions)):
        predicted_label, true_label = category_accuracy(i, predictions, test_labels)
        confusion_matrix[true_label][predicted_label] += 1
    plot_confusion_matrix(confusion_matrix, path, target_names=class_name_list, normalize=False, title='CM')
    plot_confusion_matrix(confusion_matrix, path, target_names=class_name_list, title='CM_Nomal')
    return confusion_matrix


def calc_Metric(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    print(f"FP: {FP}, FN: {FN}, TP: {TP}, TN: {TN}")

    accuracy = (TP + TN) / (FP + FN + TP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = TP / (TP + (FN + FP) / 2)

    print(f"Accuracy : {accuracy}")
    print(f"Precision : {precision}")
    print(f"Recall : {recall}")
    print(f"F1 : {F1}")

    return accuracy, precision, recall, F1


def write_result(output_list, result_df):
    result_df.loc[output_list[0]] = output_list
    return result_df


def dice(true_mask, pred_mask, non_seg_score=1.0):
    """
        Computes the Dice coefficient.
        Args:
            true_mask : Array of arbitrary shape.
            pred_mask : Array with the same shape than true_mask.

        Returns:
            A scalar representing the Dice coefficient between the two segmentations.

    """
    assert true_mask.shape == pred_mask.shape

    true_mask = np.asarray(true_mask).astype(np.bool)
    pred_mask = np.asarray(pred_mask).astype(np.bool)

    # If both segmentations are all zero, the dice will be 1. (Developer decision)
    im_sum = true_mask.sum() + pred_mask.sum()
    if im_sum == 0:
        return non_seg_score

    # Compute Dice coefficient
    intersection = np.logical_and(true_mask, pred_mask)
    return 2. * intersection.sum() / im_sum

class MeanIoU(object):
    """Mean intersection over union (mIoU) metric.
    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:
        IoU = true_positive / (true_positive + false_positive + false_negative).
    The mean IoU is the mean of IoU between all classes.
    Keyword arguments:
        num_classes (int): number of classes in the classification problem.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

    def mean_iou(self, y_true, y_pred):
        """The metric function to be passed to the model.
        Args:
            y_true (tensor): True labels.
            y_pred (tensor): Predictions of the same shape as y_true.
        Returns:
            The mean intersection over union as a tensor.
        """
        # Wraps _mean_iou function and uses it as a TensorFlow op.
        # Takes numpy arrays as its arguments and returns numpy arrays as
        # its outputs.
        return tf.py_func(self._mean_iou, [y_true, y_pred], tf.float32)

    def _mean_iou(self, y_true, y_pred):
        """Computes the mean intesection over union using numpy.
        Args:
            y_true (tensor): True labels.
            y_pred (tensor): Predictions of the same shape as y_true.
        Returns:
            The mean intersection over union (np.float32).
        """
        # Compute the confusion matrix to get the number of true positives,
        # false positives, and false negatives
        # Convert predictions and target from categorical to integer format
        target = np.argmax(y_true, axis=-1).ravel()
        predicted = np.argmax(y_pred, axis=-1).ravel()

        # Trick for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=self.num_classes**2
        )
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape(
            (self.num_classes, self.num_classes)
        )

        # Compute the IoU and mean IoU from the confusion matrix
        true_positive = np.diag(conf)
        false_positive = np.sum(conf, 0) - true_positive
        false_negative = np.sum(conf, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error and
        # set the value to 1 since we predicted 0 pixels for that class and
        # and the batch has 0 pixels for that same class
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        iou[np.isnan(iou)] = 1

        return np.mean(iou).astype(np.float32)



