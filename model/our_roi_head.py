import torch
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign, boxes as box_ops
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.roi_heads import fastrcnn_loss, maskrcnn_inference, \
    keypointrcnn_inference, keypointrcnn_loss, maskrcnn_loss
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads


class ModifiedRoIHead(RoIHeads):
    def postprocess_detections(self, box_features, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores and featyres per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_features = box_features.split(boxes_per_image, 0)
        all_boxes = []
        all_scores = []
        all_labels = []
        all_features = []
        for features, boxes, scores, image_shape in zip(pred_features, pred_boxes, pred_scores, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()

            # at this stage every feature correspond to nClasses-1 boxes
            # so the idea is to keep track of which original box indices are kept
            # and from them compute feature_indices = box_idx // nClasses-1
            box_keep_idxs = torch.arange(boxes.shape[0])

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            if boxes[inds].shape[0] > 0:
                boxes, scores, labels, box_keep_idxs = boxes[inds], scores[inds], labels[inds], box_keep_idxs[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            if boxes[keep].shape[0] > 0:
                boxes, scores, labels, box_keep_idxs = boxes[keep], scores[keep], labels[keep], box_keep_idxs[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            if boxes[keep].shape[0] > 0:
                boxes, scores, labels, box_keep_idxs = boxes[keep], scores[keep], labels[keep], box_keep_idxs[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

            feature_indices = box_keep_idxs / (num_classes-1)
            all_features.append(features[feature_indices])
        return all_boxes, all_scores, all_labels, all_features

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint:
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(
                proposals, targets)

        # this is where we changed the code so that boxes will be returned in training too
        box_features_return = self.box_roi_pool(features, proposals,
                                                image_shapes)
        box_features = self.box_head(box_features_return)
        class_logits, box_regression = self.box_predictor(box_features)
        result, losses = [], {}

        if self.training:
            boxes, scores, new_labels, GCN_features = self.postprocess_detections(box_features_return, class_logits,
                                                                                  box_regression, proposals,
                                                                                  image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=new_labels[i],
                        scores=scores[i],
                    )
                )

            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = dict(loss_classifier=loss_classifier,
                          loss_box_reg=loss_box_reg)
        else:
            boxes, scores, labels, GCN_features = self.postprocess_detections(box_features_return, class_logits,
                                                                              box_regression, proposals,
                                                                              image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                    )
                )

        if self.has_mask:
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])

            mask_features = self.mask_roi_pool(
                features, mask_proposals, image_shapes)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)

            loss_mask = {}
            if self.training:
                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = dict(loss_mask=loss_mask)
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        if self.has_keypoint:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])

            keypoint_features = self.keypoint_roi_pool(
                features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                gt_keypoints = [t["keypoints"] for t in targets]
                loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = dict(loss_keypoint=loss_keypoint)
            else:
                keypoints_probs, kp_scores = keypointrcnn_inference(
                    keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses, GCN_features


def build_RoI_head(out_channels, num_classes=None, box_roi_pool=None, box_head=None, box_predictor=None,
                   box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                   box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                   box_batch_size_per_image=512, box_positive_fraction=0.25,
                   bbox_reg_weights=None, mask_predictor=None, mask_roi_pool=None, mask_head=None):
    if box_roi_pool is None:
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=7,
            sampling_ratio=2)

    if box_head is None:
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

    if box_predictor is None:
        representation_size = 1024
        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes)

    if mask_roi_pool is None:
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=14,
            sampling_ratio=2)

    if mask_head is None:
        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

    roi_heads = ModifiedRoIHead(
        # Box
        box_roi_pool, box_head, box_predictor,
        box_fg_iou_thresh, box_bg_iou_thresh,
        box_batch_size_per_image, box_positive_fraction,
        bbox_reg_weights,
        box_score_thresh, box_nms_thresh, box_detections_per_img, mask_predictor=mask_predictor, mask_head=mask_head,
        mask_roi_pool=mask_roi_pool)
    return roi_heads
