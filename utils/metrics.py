import torch
from torch import Tensor


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


def batch_iou(a: Tensor, b: Tensor):
    """ Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        a: Tensor matrix where each row containing [x1,y1,x2,y2] coordinates
        b: Tensor matrix where each row containing [x1,y1,x2,y2] coordinates

    Returns:
        Tensor The Intersect of Union scores for each pair of bounding
        boxes.
    """
    # COORDINATES OF THE INTERSECTION BOXES

    x1 = torch.max(a[:, 0], b[:, 0])
    y1 = torch.max(a[:, 1], b[:, 1])
    x2 = torch.max(a[:, 2], b[:, 2])
    y2 = torch.max(a[:, 3], b[:, 3])

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + 1e-8)
    return iou
