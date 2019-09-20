import torch
from torch import Tensor
from torchvision.ops.boxes import box_iou


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

    class_precision = 100*(tp/total_positive_predicted)
    class_recall = 100*(tp/should_be_positive)

    score = (1+beta**2)*class_precision*class_recall

    return score / (1e-8+class_recall+(beta**2)*class_precision)
