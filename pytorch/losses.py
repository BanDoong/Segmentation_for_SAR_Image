import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice


# PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


def dice_coef_torch(y_true, y_pred, const=1e-07):
    # flatten 2-d tensors
    y_true_pos = y_true.view(-1)
    y_pred_pos = y_pred.view(-1)

    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)

    # 2TP/(2TP+FP+FN) == 2TP/()
    coef_val = (2.0 * true_pos + const) / (2.0 * true_pos + false_pos + false_neg)

    return coef_val


def pix_acc_torch(y_true, y_pred):
    # flatten 2-d tensors
    y_true_pos = y_true.view(-1)
    y_pred_pos = y_pred.view(-1)

    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    true_neg = torch.sum((1 - y_true_pos) * (1 - y_pred_pos))
    pix_acc = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    return pix_acc


def iou_seg_torch(y_true, y_pred):
    # tf tensor casting
    # y_pred = tf.convert_to_tensor(y_pred)
    # y_pred = tf.cast(y_pred, dtype)
    # y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = y_pred.type(torch.FloatTensor)
    y_true = y_true.type(torch.FloatTensor)

    y_pred = torch.squeeze(y_pred)
    y_true = torch.squeeze(y_true)

    y_true_pos = y_true.view(-1)
    y_pred_pos = y_pred.view(-1)

    area_intersect = torch.sum(torch.multiply(y_true_pos, y_pred_pos))

    area_true = torch.sum(y_true_pos)
    area_pred = torch.sum(y_pred_pos)
    area_union = area_true + area_pred - area_intersect
    if area_union <= 0:
        raise "Area of Union is under ZERO. You can't devide with zero."
    iou = area_intersect / area_union
    return iou


ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


# PyTorch
ALPHA = 0.5  # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5  # weighted contribution of modified CE loss compared to Dice loss

"""
"Combo loss: Handling input and output imbalance in multi-organ segmentation",
retrievable here: https://arxiv.org/abs/1805.02798. 
"""

"""
class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, eps=1e-9):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)

        return combo
"""
