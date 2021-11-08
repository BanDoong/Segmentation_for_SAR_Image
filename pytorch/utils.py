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


def make_df(num_modality, modality):
    if num_modality == 3:
        result_df = pd.DataFrame(
            {'epoch': [], 'train_accuracy': [], 'val_accuracy': [],
             'train_classification_loss': [], 'train_mri_loss': [], 'train_tau_loss': [],
             'train_amyloid_loss': [],
             'val_classification_loss': [], 'val_mri_loss': [], 'val_tau_loss': [], 'val_amyloid_loss': []})

    elif num_modality == 2:
        if 'mri' and 'tau' in modality:
            result_df = pd.DataFrame(
                {'epoch': [], 'train_accuracy': [], 'val_accuracy': [],
                 'train_classification_loss': [], 'train_mri_loss': [], 'train_tau_loss': [],
                 'val_classification_loss': [], 'val_mri_loss': [], 'val_tau_loss': []})
        elif 'mri' and 'amyloid' in modality:
            result_df = pd.DataFrame(
                {'epoch': [], 'train_accuracy': [], 'val_accuracy': [],
                 'train_classification_loss': [], 'train_mri_loss': [], 'train_amyloid_loss': [],
                 'val_classification_loss': [], 'val_mri_loss': [], 'val_amyloid_loss': []})
    else:
        result_df = pd.DataFrame(
            {'epoch': [], 'train_accuracy': [], 'val_accuracy': [],
             'train_classification_loss': [], 'train_mri_loss': [], 'val_classification_loss': [],
             'val_mri_loss': []})
    return result_df


