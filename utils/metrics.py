import torch
from torch import Tensor
from torchvision.ops.boxes import box_iou
from sklearn.metrics import auc


def f_score(confusion: Tensor, beta=1.0):
    '''
        computes the general F score based on the given confusion matrix.

        confusion: a n_class X n_class matrix where c[i,j]=number of class i predictions
        where made that according to the ground truth should have been class j

        beta: an averging coeeficient berween precision and recall

    '''
    # precision is number of true class predictions / total class prediction
    # recall is number of true class predictions / number of class in gt
    tp = confusion.diagonal()
    should_be_positive = confusion.sum(0)
    total_positive_predicted = confusion.sum(1)

    class_precision = 100 * (tp / total_positive_predicted)
    class_recall = 100 * (tp / should_be_positive)

    score = (1 + beta ** 2) * class_precision * class_recall

    return score / (1e-8 + class_recall + (beta ** 2) * class_precision)


def calc_precision_box(boxes, gt_boxes):
    count = 0
    num_sampels = len(boxes)

    for gt_box, pred_box in zip(gt_boxes, boxes):
        if box_iou(gt_box, pred_box)[0][0] > 0.5:
            count += 1
    return count / num_sampels


def calc_precision_mask(masks, gt_masks):
    count = 0
    num_sampels = len(masks)

    for mask, gt_mask in zip(masks, gt_masks):
        intersection = mask & gt_mask
        union = mask | gt_mask
        iou_score = torch.sum(intersection) / torch.sum(union)
        if iou_score > 0.5:
            count += 1
    return count / num_sampels


def mesh_precision_recall(confusion, f1_score):
    tp = confusion.diagonal()
    should_be_positive = confusion.sum(0)
    total_positive_predicted = confusion.sum(1)
    tp[f1_score <= 0.5] = 0  # at f1_0.3 > 0.5 condition for being true positive
    class_precision = 100 * (tp / total_positive_predicted)
    class_recall = 100 * (tp / should_be_positive)
    return auc(class_recall, class_precision)
