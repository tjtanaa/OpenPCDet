import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


def compute_fg_mask(gt_boxes2d, shape, downsample_factor=1, device=torch.device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_boxes2d: (B, N, 4), 2D box labels
        shape: torch.Size or tuple, Foreground mask desired shape
        downsample_factor: int, Downsample factor for image
        device: torch.device, Foreground mask desired device
    Returns:
        fg_mask (shape), Foreground mask
    """
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :, :2] = torch.floor(gt_boxes2d[:, :, :2])
    gt_boxes2d[:, :, 2:] = torch.ceil(gt_boxes2d[:, :, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    B, N = gt_boxes2d.shape[:2]
    for b in range(B):
        for n in range(N):
            u1, v1, u2, v2 = gt_boxes2d[b, n]
            fg_mask[b, v1:v2, u1:u2] = True

    return fg_mask


def neg_loss_cornernet(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self):
        super(FocalLossCenterNet, self).__init__()
        self.neg_loss = neg_loss_cornernet

    def forward(self, out, target, mask=None):
        return self.neg_loss(out, target, mask=mask)


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    isnotnan = (~ torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = torch.abs(regr - gt_regr)
    loss = loss.transpose(2, 0)

    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    # else:
    #  # D x M x B
    #  loss = loss.reshape(loss.shape[0], -1)

    # loss = loss / (num + 1e-4)
    loss = loss / torch.clamp_min(num, min=1.0)
    # import pdb; pdb.set_trace()
    return loss


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossCenterNet, self).__init__()

    def forward(self, output, mask, ind=None, target=None):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
        else:
            pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss




class IoU3DLossVariablePointHead(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        
        self.eps = torch.constant(1e-6)
        
    def _roi_logits_to_attrs(self, base_coors, input_logits, anchor_size):
        anchor_diag = torch.sqrt(torch.pow(anchor_size[0], 2.) + torch.pow(anchor_size[1], 2.))
        x = torch.clamp(input_logits[:, 0] * anchor_diag + base_coors[:, 0], -1e7, 1e7)
        y = torch.clamp(input_logits[:, 1] * anchor_diag + base_coors[:, 1], -1e7, 1e7)
        z = torch.clamp(input_logits[:, 2] * anchor_diag + base_coors[:, 2], -1e7, 1e7)

        w = torch.clamp(torch.exp(input_logits[:,3]) * anchor_size[0], 0., 1e7 )
        l = torch.clamp(torch.exp(input_logits[:,4]) * anchor_size[1], 0., 1e7 )
        h = torch.clamp(torch.exp(input_logits[:,5]) * anchor_size[2], 0., 1e7 )

        r = torch.clamp(torch.atan2(input_logits[:,6], input_logits[:,7]), -1e7, 1e7)

        return torch.stack([w, l, h, x, y, z, r], axis=-1)

    def _get_rotation_matrix(self, r):
        rotation_matrix = torch.stack([torch.cos(r), -torch.sin(r), torch.sin(r), torch.cos(r)], axis=-1)
        rotation_matrix = torch.reshape(rotation_matrix, shape=[-1, 2, 2]) # [n, 2, 2]
        return rotation_matrix

    def _get_2d_vertex_points(self, gt_attrs, pred_attrs):
        gt_w = gt_attrs[:, 0]  # [n]
        gt_l = gt_attrs[:, 1]  # [n]
        gt_x = gt_attrs[:, 3]  # [n]
        gt_y = gt_attrs[:, 4]  # [n]
        gt_r = gt_attrs[:, 6]  # [n]

        gt_v0 = torch.stack([gt_w / 2, -gt_l / 2], axis=-1)  # [n, 2]
        gt_v1 = torch.stack([gt_w / 2, gt_l / 2], axis=-1)  # [n, 2]
        gt_v2 = torch.stack([-gt_w / 2, gt_l / 2], axis=-1)  # [n, 2]
        gt_v3 = torch.stack([-gt_w / 2, -gt_l / 2], axis=-1)  # [n, 2]
        gt_v = torch.stack([gt_v0, gt_v1, gt_v2, gt_v3], axis=1)  # [n, 4, 2]

        pred_w = pred_attrs[:, 0]  # [n]
        pred_l = pred_attrs[:, 1]  # [n]
        pred_x = pred_attrs[:, 3]  # [n]
        pred_y = pred_attrs[:, 4]  # [n]
        pred_r = pred_attrs[:, 6]  # [n]

        rel_x = pred_x - gt_x  # [n]
        rel_y = pred_y - gt_y  # [n]
        rel_r = pred_r - gt_r  # [n]
        rel_xy = torch.unsqueeze(torch.stack([rel_x, rel_y], axis=-1), axis=1)  # [n, 1, 2]

        pred_v0 = torch.stack([pred_w / 2, -pred_l / 2], axis=-1)  # [n, 2]
        pred_v1 = torch.stack([pred_w / 2, pred_l / 2], axis=-1)  # [n, 2]
        pred_v2 = torch.stack([-pred_w / 2, pred_l / 2], axis=-1)  # [n, 2]
        pred_v3 = torch.stack([-pred_w / 2, -pred_l / 2], axis=-1)  # [n, 2]
        pred_v = torch.stack([pred_v0, pred_v1, pred_v2, pred_v3], axis=1)  # [n, 4, 2]

        rot_pred_v = torch.transpose(torch.matmul(a=self.get_rotation_matrix(rel_r), b=pred_v, transpose_b=True),
                                perm=[0, 2, 1])  # [n, 4, 2]
        rot_rel_xy = torch.transpose(torch.matmul(a=self.get_rotation_matrix(-gt_r), b=rel_xy, transpose_b=True),
                                perm=[0, 2, 1])  # [n, 1, 2]
        rel_rot_pred_v = rot_pred_v + rot_rel_xy  # [n, 4, 2]

        rot_gt_v = torch.transpose(torch.matmul(a=self.get_rotation_matrix(-rel_r), b=gt_v, transpose_b=True),
                                perm=[0, 2, 1])  # [n, 4, 2]
        rot_rel_xy = torch.transpose(torch.matmul(a=self.get_rotation_matrix(-pred_r), b=-rel_xy, transpose_b=True),
                                perm=[0, 2, 1])  # [n, 1, 2]
        rel_rot_gt_v = rot_gt_v + rot_rel_xy  # [n, 4, 2]

        # [n, 2, 2] @ [n, 2, 4] = [n, 2, 4] -> [n, 4, 2]

        return gt_v, rel_rot_pred_v, rel_rot_gt_v, rel_xy, rel_r


    def _get_2d_intersection_points(self, gt_attrs, rel_rot_pred_v):
        gt_w = gt_attrs[:, 0]  # [n]
        gt_l = gt_attrs[:, 1]  # [n]
        output_points = []
        for i in [-1, 0, 1, 2]:
            v0_x = rel_rot_pred_v[:, i, 0]  # [n]
            v0_y = rel_rot_pred_v[:, i, 1]  # [n]
            v1_x = rel_rot_pred_v[:, i + 1, 0]  # [n]
            v1_y = rel_rot_pred_v[:, i + 1, 1]  # [n]

            kx = torch.nan_to_num(torch.div(v1_y - v0_y, v1_x - v0_x))
            bx = torch.nan_to_num(torch.div(v0_y * v1_x - v1_y * v0_x, v1_x - v0_x))
            ky = torch.nan_to_num(torch.div(v1_x - v0_x, v1_y - v0_y))
            by = torch.nan_to_num(torch.div(v1_y * v0_x - v0_y * v1_x, v1_y - v0_y))

            # kx = (v1_y - v0_y) / (v1_x - v0_x + eps) # [n]
            # bx = (v0_y * v1_x - v1_y * v0_x) / (v1_x - v0_x + eps) # [n]
            # ky = (v1_x - v0_x) / (v1_y - v0_y + eps) # [n]
            # by = (v1_y * v0_x - v0_y * v1_x) / (v1_y - v0_y + eps) # [n]

            p0 = torch.stack([gt_w / 2, kx * gt_w / 2 + bx], axis=-1)  # [n, 2]
            p1 = torch.stack([-gt_w / 2, -kx * gt_w / 2 + bx], axis=-1)  # [n, 2]
            p2 = torch.stack([ky * gt_l / 2 + by, gt_l / 2], axis=-1)  # [n, 2]
            p3 = torch.stack([-ky * gt_l / 2 + by, -gt_l / 2], axis=-1)  # [n, 2]
            p = torch.stack([p0, p1, p2, p3], axis=1)  # [n, 4, 2]
            output_points.append(p)
        output_points = torch.concat(output_points, axis=1)  # [n, 16, 2]
        return output_points


    def _get_interior_vertex_points_mask(self, target_attrs, input_points):
        target_w = torch.unsqueeze(target_attrs[:, 0], axis=1)  # [n, 1, 16]
        target_l = torch.unsqueeze(target_attrs[:, 1], axis=1)  # [n, 1, 16]
        target_x = target_w / 2  # [n, 4]
        target_y = target_l / 2  # [n, 4]
        x_mask = torch.le(torch.abs(input_points[:, :, 0]), target_x).type(torch.float32)  # [n, 4]
        y_mask = torch.le(torch.abs(input_points[:, :, 1]), target_y).type(torch.float32)   # [n, 4]
        return x_mask * y_mask  # [n, 4]

    def _get_intersection_points_mask(self, target_attrs, input_points, rel_xy=None, rel_r=None):
        if rel_xy is not None and rel_r is not None:
            pred_r = target_attrs[:, 6]  # [n]
            rot_input_points = torch.transpose(torch.matmul(a=self.get_rotation_matrix(-rel_r), b=input_points, transpose_b=True),
                                            perm=[0, 2, 1])  # [n, 16, 2]
            rot_rel_xy = torch.transpose(torch.matmul(a=self.get_rotation_matrix(-pred_r), b=-rel_xy, transpose_b=True),
                                    perm=[0, 2, 1])  # [n, 1, 2]
            rel_rot_input_points = rot_input_points + rot_rel_xy
        else:
            rel_rot_input_points = input_points
        target_w = torch.unsqueeze(target_attrs[:, 0], axis=1)  # [n, 1, 16]
        target_l = torch.unsqueeze(target_attrs[:, 1], axis=1)  # [n, 1, 16]
        target_x = target_w / 2 + 1e-3  # [n, 4]
        target_y = target_l / 2 + 1e-3  # [n, 4]
        # target_x = 1000  # [n, 4]
        # target_y = 1000  # [n, 4]
        max_x_mask = torch.le(torch.abs(rel_rot_input_points[:, :, 0]), target_x).type(torch.float32)  # [n, 4]
        max_y_mask = torch.le(torch.abs(rel_rot_input_points[:, :, 1]), target_y).type(torch.float32)  # [n, 4]
        return max_x_mask * max_y_mask  # [n, 4]


    def _clockwise_sorting(self, input_points, masks):
        coors_masks = torch.stack([masks, masks], axis=-1)  # [n, 24, 2]
        masked_points = input_points * coors_masks
        centers = torch.nan_to_num(torch.div(torch.sum(masked_points, axis=1, keepdim=True),
                                        (torch.sum(coors_masks, axis=1, keepdim=True))))  # [n, 1, 2]
        rel_vectors = input_points - centers  # [n, 24, 2]
        base_vector = rel_vectors[:, :1, :]  # [n, 1, 2]
        # https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors/16544330#16544330
        dot = base_vector[:, :, 0] * rel_vectors[:, :, 0] + base_vector[:, :, 1] * rel_vectors[:, :, 1]  # [n, 24]
        det = base_vector[:, :, 0] * rel_vectors[:, :, 1] - base_vector[:, :, 1] * rel_vectors[:, :, 0]  # [n, 24]
        angles = torch.atan2(det + self.eps, dot + self.eps)  # [n, 24] -pi~pi
        angles_masks = (0.5 - (masks - 0.5)) * 1000.  # [n, 24]
        masked_angles = angles + angles_masks  # [n, 24]
        _, sort_idx = torch.topk(-masked_angles, k=input_points.get_shape().as_list()[1], sorted=True)  # [n, 24]

        batch_id = torch.range(start=0, limit=input_points.shape[0], dtype=torch.int32)
        batch_ids = torch.stack([batch_id] * input_points.get_shape().as_list()[1], axis=1)
        sort_idx = torch.stack([batch_ids, sort_idx], axis=-1)  # [n, 24, 2]

        
        sorted_points = input_points.index_select(0, sort_idx)
        sorted_masks = masks.index_select(0, sort_idx)

        # sorted_points = tf.gather_nd(input_points, sort_idx)
        # sorted_masks = tf.gather_nd(masks, sort_idx)

        return sorted_points, sorted_masks



    def _shoelace_intersection_area(self, sorted_points, sorted_masks):
        # https://en.wikipedia.org/wiki/Shoelace_formula
        sorted_points = sorted_points * torch.stack([sorted_masks, sorted_masks], axis=-1)  # [n, 24, 2]
        last_vertex_id = (torch.sum(sorted_masks, axis=1) - 1).type(torch.int32)  # [n] coors where idx=-1 will be convert to [0., 0.], so it's safe.
        last_vertex_id = torch.stack([torch.range(start=0, limit=sorted_points.shape[0], dtype=torch.int32), last_vertex_id],
                                axis=-1)  # [n, 2]
        last_vertex_to_duplicate = torch.unsqueeze(sorted_points.index_select(0, last_vertex_id), axis=1)  # [n, 1, 2]
        padded_sorted_points = torch.cat([last_vertex_to_duplicate, sorted_points], axis=1)  # [n, 24+1, 2]
        x_i = padded_sorted_points[:, :-1, 0]  # [n, 24]
        x_i_plus_1 = padded_sorted_points[:, 1:, 0]  # [n, 24]
        y_i = padded_sorted_points[:, :-1, 1]  # [n, 24]
        y_i_plus_1 = padded_sorted_points[:, 1:, 1]  # [n, 24]
        area = 0.5 * torch.sum(x_i * y_i_plus_1 - x_i_plus_1 * y_i, axis=-1)  # [n]
        return area


    def _get_intersection_height(self, gt_attrs, pred_attrs):
        gt_h = gt_attrs[:, 2]
        gt_z = gt_attrs[:, 5]
        pred_h = pred_attrs[:, 2]
        pred_z = pred_attrs[:, 5]
        gt_low = gt_z - 0.5 * gt_h
        gt_high = gt_z + 0.5 * gt_h
        pred_low = pred_z - 0.5 * pred_h
        pred_high = pred_z + 0.5 * pred_h
        top = torch.minimum(gt_high, pred_high)
        bottom = torch.maximum(gt_low, pred_low)
        intersection_height = F.ReLU(top - bottom)
        return intersection_height


    def _get_3d_iou_from_area(self, gt_attrs, pred_attrs, intersection_2d_area, intersection_height, clip):
        intersection_volume = intersection_2d_area * intersection_height
        gt_volume = gt_attrs[:, 0] * gt_attrs[:, 1] * gt_attrs[:, 2]
        pred_volume = pred_attrs[:, 0] * pred_attrs[:, 1] * pred_attrs[:, 2]
        iou = torch.nan_to_num(torch.div(intersection_volume, gt_volume + pred_volume - intersection_volume))
        # tf.summary.scalar('iou_nan_sum',
        #                   hvd.allreduce(tf.reduce_sum(tf.cast(tf.is_nan(iou), dtype=tf.float32)), average=False))
        if clip:
            iou = torch.where(torch.is_nan(iou), torch.zeros_like(iou), iou)
        return iou



    def _cal_3d_iou(self, gt_attrs, pred_attrs, clip=False):
        gt_v, rel_rot_pred_v, rel_rot_gt_v, rel_xy, rel_r = self._get_2d_vertex_points(gt_attrs, pred_attrs)
        intersection_points = self._get_2d_intersection_points(gt_attrs=gt_attrs, rel_rot_pred_v=rel_rot_pred_v)
        gt_vertex_points_inside_pred = self._get_interior_vertex_points_mask(target_attrs=pred_attrs, input_points=rel_rot_gt_v)
        pred_vertex_points_inside_gt = self._get_interior_vertex_points_mask(target_attrs=gt_attrs, input_points=rel_rot_pred_v)
        pred_intersect_with_gt = self._get_intersection_points_mask(target_attrs=gt_attrs, input_points=intersection_points)
        intersection_points_inside_pred = self._get_intersection_points_mask(target_attrs=pred_attrs,
                                                                    input_points=intersection_points, rel_xy=rel_xy,
                                                                    rel_r=rel_r)
        total_points = torch.cat([gt_v, rel_rot_pred_v, intersection_points], axis=1)
        total_masks = torch.cat([gt_vertex_points_inside_pred, pred_vertex_points_inside_gt,
                                pred_intersect_with_gt * intersection_points_inside_pred], axis=1)
        sorted_points, sorted_masks = self._clockwise_sorting(input_points=total_points, masks=total_masks)

        intersection_2d_area = self._shoelace_intersection_area(sorted_points, sorted_masks)
        intersection_height = self._get_intersection_height(gt_attrs, pred_attrs)
        ious = self._get_3d_iou_from_area(gt_attrs, pred_attrs, intersection_2d_area, intersection_height, clip)

        return ious

    def forward(self):
        pass



