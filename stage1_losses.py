"""
stage1_losses.py — SEGUE Stage 1 损失函数
==========================================
L_geom = L1(bbox) + GIoU(bbox) + L2(keypoints)
L_align = Dice(mask) + BCE(mask)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, generalized_box_iou

from stage1_config import *


# ============================================================
# 1. BBox 损失: L1 + GIoU
# ============================================================
class BBoxLoss(nn.Module):
    """
    计算预测 bbox 和 GT bbox 之间的 L1 + GIoU 损失。

    输入格式: cxcywh 归一化坐标
    """

    def __init__(self, l1_weight=L1_WEIGHT, giou_weight=GIOU_WEIGHT):
        super().__init__()
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight

    def forward(self, pred_boxes, gt_boxes):
        """
        Args:
            pred_boxes: [N, 4] 预测 bbox (cxcywh 归一化)
            gt_boxes:   [N, 4] GT bbox (cxcywh 归一化)

        Returns:
            loss_bbox: scalar
            loss_dict: {'l1': ..., 'giou': ...}
        """
        if len(pred_boxes) == 0:
            return torch.tensor(0.0, device=pred_boxes.device), {'l1': 0.0, 'giou': 0.0}

        # L1 loss
        loss_l1 = F.l1_loss(pred_boxes, gt_boxes)

        # GIoU loss: 需要 xyxy 格式
        pred_xyxy = box_convert(pred_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        gt_xyxy = box_convert(gt_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        giou = generalized_box_iou(pred_xyxy, gt_xyxy)  # [N], range [-1, 1]
        loss_giou = (1.0 - giou).mean()

        loss = self.l1_weight * loss_l1 + self.giou_weight * loss_giou

        return loss, {
            'l1': loss_l1.detach().item(),
            'giou': loss_giou.detach().item(),
        }


# ============================================================
# 2. 关键点损失: L2
# ============================================================
class PointLoss(nn.Module):
    """
    计算预测关键点和 GT 关键点之间的 L2 损失。

    GT 关键点从 GT mask 中采样：取 mask 的质心 + 离质心最远的点
    """

    def __init__(self, point_weight=POINT_WEIGHT):
        super().__init__()
        self.point_weight = point_weight

    def forward(self, pred_points, gt_points):
        """
        Args:
            pred_points: [N, K, 2] 预测关键点 (像素坐标)
            gt_points:   [N, K, 2] GT 关键点 (像素坐标)

        Returns:
            loss_point: scalar
        """
        if len(pred_points) == 0:
            return torch.tensor(0.0, device=pred_points.device)

        loss = F.mse_loss(pred_points, gt_points)
        return self.point_weight * loss


# ============================================================
# 3. Mask 损失: Dice + BCE
# ============================================================
class MaskLoss(nn.Module):
    """
    Dice loss + Binary Cross Entropy loss
    对 SAM2 输出的 mask 进行监督
    """

    def __init__(self, dice_weight=DICE_WEIGHT, bce_weight=BCE_WEIGHT, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, pred_masks, gt_masks):
        """
        Args:
            pred_masks: [N, H, W] 预测 mask logits
            gt_masks:   [N, H, W] GT mask (0/1)

        Returns:
            loss_mask: scalar
            loss_dict: {'dice': ..., 'bce': ...}
        """
        if len(pred_masks) == 0:
            return torch.tensor(0.0, device=pred_masks.device), {'dice': 0.0, 'bce': 0.0}

        pred_flat = pred_masks.reshape(-1)
        gt_flat = gt_masks.reshape(-1).float()

        # BCE loss (with logits)
        loss_bce = F.binary_cross_entropy_with_logits(pred_flat, gt_flat)

        # Dice loss
        pred_sigmoid = pred_masks.sigmoid().reshape(-1)
        intersection = (pred_sigmoid * gt_flat).sum()
        union = pred_sigmoid.sum() + gt_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss_dice = 1.0 - dice

        loss = self.dice_weight * loss_dice + self.bce_weight * loss_bce

        return loss, {
            'dice': loss_dice.detach().item(),
            'bce': loss_bce.detach().item(),
        }


# ============================================================
# 4. 综合 Grounding Loss
# ============================================================
class GroundingLoss(nn.Module):
    """
    Stage 1 总损失: L_geom + L_align

    L_geom = L1(bbox) + GIoU(bbox) + L2(keypoints)
    L_align = Dice(mask) + BCE(mask)
    """

    def __init__(self):
        super().__init__()
        self.bbox_loss = BBoxLoss()
        self.point_loss = PointLoss()
        self.mask_loss = MaskLoss()

    def forward(self, outputs, targets, compute_mask_loss=True):
        """compute_mask_loss=False 时只计算 L_geom (跳过 SAM2)"""
        B = len(outputs['pred_bboxes'])

        total_loss = torch.tensor(0.0, device=next(iter(
            outputs['pred_bboxes'][0] if len(outputs['pred_bboxes'][0]) > 0
            else torch.zeros(1)
        )).device)

        all_bbox_loss = 0.0
        all_point_loss = 0.0
        all_mask_loss = 0.0
        valid_count = 0

        for b in range(B):
            pred_bboxes = outputs['pred_bboxes'][b]
            pred_points = outputs['pred_keypoints'][b]
            pred_masks = outputs['pred_masks'][b]
            gt_bboxes = targets['gt_bboxes'][b]
            gt_points = targets['gt_points'][b]
            gt_masks = targets['gt_masks'][b]

            if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
                continue

            valid_count += 1

            # ---- Matching: 找到 pred 和 GT 之间的对应关系 ----
            # 使用 IoU 匹配（bbox 级别）
            match_indices = self._match_by_bbox_iou(pred_bboxes, gt_bboxes)

            if len(match_indices) == 0:
                continue

            pred_idx = match_indices[:, 0]
            gt_idx = match_indices[:, 1]

            matched_pred_bboxes = pred_bboxes[pred_idx]
            matched_gt_bboxes = gt_bboxes[gt_idx]
            matched_pred_points = pred_points[pred_idx]
            matched_gt_points = gt_points[gt_idx]
            matched_pred_masks = pred_masks[pred_idx]
            matched_gt_masks = gt_masks[gt_idx]

            # BBox loss
            b_loss, b_dict = self.bbox_loss(matched_pred_bboxes, matched_gt_bboxes)
            total_loss = total_loss + b_loss
            all_bbox_loss += b_dict['l1'] + b_dict['giou']

            # Point loss
            p_loss = self.point_loss(matched_pred_points, matched_gt_points)
            total_loss = total_loss + p_loss
            all_point_loss += p_loss.detach().item() if torch.is_tensor(p_loss) else p_loss

            # Mask loss (skip during training without SAM2)
            if compute_mask_loss:
                m_loss, m_dict = self.mask_loss(matched_pred_masks, matched_gt_masks)
                total_loss = total_loss + m_loss
                all_mask_loss += m_dict['dice'] + m_dict['bce']

        if valid_count > 0:
            total_loss = total_loss / valid_count
            all_bbox_loss /= valid_count
            all_point_loss /= valid_count
            all_mask_loss /= valid_count

        loss_dict = {
            'total': total_loss.detach().item() if torch.is_tensor(total_loss) else total_loss,
            'bbox': all_bbox_loss,
            'point': all_point_loss,
            'mask': all_mask_loss,
        }

        return total_loss, loss_dict

    def _match_by_bbox_iou(self, pred_boxes, gt_boxes, iou_threshold=0.3):
        """
        贪心匹配 pred bbox 和 GT bbox。

        Args:
            pred_boxes: [N, 4] cxcywh 归一化
            gt_boxes:   [M, 4] cxcywh 归一化
            iou_threshold: 最低 IoU 阈值

        Returns:
            matches: [K, 2] tensor of (pred_idx, gt_idx)
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return torch.empty(0, 2, dtype=torch.long, device=pred_boxes.device)

        pred_xyxy = box_convert(pred_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        gt_xyxy = box_convert(gt_boxes, in_fmt='cxcywh', out_fmt='xyxy')

        # 计算 pairwise IoU
        # 使用 box_iou
        from torchvision.ops import box_iou
        iou_matrix = box_iou(pred_xyxy, gt_xyxy)  # [N, M]

        # 贪心匹配: 对每个 GT 找最佳 pred
        matches = []
        gt_matched = set()

        for gt_j in range(len(gt_boxes)):
            if len(pred_boxes) == 0:
                break
            ious = iou_matrix[:, gt_j]
            best_pred = ious.argmax().item()
            if ious[best_pred] > iou_threshold and best_pred not in [m[0] for m in matches]:
                matches.append((best_pred, gt_j))

        if matches:
            return torch.tensor(matches, dtype=torch.long, device=pred_boxes.device)
        return torch.empty(0, 2, dtype=torch.long, device=pred_boxes.device)


# ============================================================
# 5. 辅助函数: 从 GT mask 生成 GT bbox 和关键点
# ============================================================
def masks_to_bboxes_and_points(masks, image_size=None):
    """
    从 binary mask 生成 bbox 和关键点。

    Args:
        masks: [N, H, W] binary masks (0/1 或 bool)
        image_size: (H, W) 如果 mask 需要 resize

    Returns:
        bboxes: [N, 4] cxcywh 归一化
        points: [N, 2, 2] 关键点 (质心 + 离质心最远的点)，像素坐标
    """
    N, H, W = masks.shape
    bboxes = []
    points_list = []

    for i in range(N):
        mask_i = masks[i].cpu().numpy()

        # 找 mask 区域的像素坐标
        ys, xs = np.where(mask_i > 0.5)
        if len(ys) == 0:
            # 空 mask → dummy
            bboxes.append(torch.tensor([0.5, 0.5, 0.01, 0.01]))
            points_list.append(torch.tensor([[H // 2, W // 2], [H // 2, W // 2]], dtype=torch.float32))
            continue

        # BBox
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + y2) / 2.0 / H
        bw = max((x2 - x1) / W, 0.01)
        bh = max((y2 - y1) / H, 0.01)
        bboxes.append(torch.tensor([cx, cy, bw, bh]))

        # 关键点: 质心 + 离质心最远的点
        centroid_x = xs.mean()
        centroid_y = ys.mean()
        # 找离质心最远的 mask 点
        dists = np.sqrt((xs - centroid_x) ** 2 + (ys - centroid_y) ** 2)
        farthest_idx = np.argmax(dists)

        pt1 = np.array([centroid_x, centroid_y])
        pt2 = np.array([xs[farthest_idx], ys[farthest_idx]])

        points_list.append(torch.from_numpy(np.stack([pt1, pt2], axis=0)).float())

    bboxes = torch.stack(bboxes)
    points = torch.stack(points_list)  # [N, 2, 2]

    return bboxes, points


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("Testing losses...")

    # 伪造数据
    pred_bboxes = torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.2, 0.2, 0.2, 0.2]])
    gt_bboxes = torch.tensor([[0.5, 0.5, 0.35, 0.32], [0.22, 0.18, 0.19, 0.21]])
    pred_points = torch.randn(2, 2, 2)
    gt_points = torch.randn(2, 2, 2)
    pred_masks = torch.randn(2, 64, 64)
    gt_masks = (torch.rand(2, 64, 64) > 0.7).float()

    # BBox loss
    bbox_loss_fn = BBoxLoss()
    loss, d = bbox_loss_fn(pred_bboxes, gt_bboxes)
    print(f"BBox loss: {loss.item():.4f}, l1={d['l1']:.4f}, giou={d['giou']:.4f}")

    # Point loss
    point_loss_fn = PointLoss()
    loss = point_loss_fn(pred_points, gt_points)
    print(f"Point loss: {loss.item():.4f}")

    # Mask loss
    mask_loss_fn = MaskLoss()
    loss, d = mask_loss_fn(pred_masks, gt_masks)
    print(f"Mask loss: {loss.item():.4f}, dice={d['dice']:.4f}, bce={d['bce']:.4f}")

    # Full Grounding Loss
    grounding_loss_fn = GroundingLoss()
    outputs = {
        'pred_bboxes': [pred_bboxes],
        'pred_keypoints': [pred_points],
        'pred_masks': [pred_masks],
    }
    targets = {
        'gt_bboxes': [gt_bboxes],
        'gt_points': [gt_points],
        'gt_masks': [gt_masks],
    }
    total, d = grounding_loss_fn(outputs, targets)
    print(f"Total loss: {total.item():.4f}, bbox={d['bbox']:.4f}, point={d['point']:.4f}, mask={d['mask']:.4f}")

    print("\n✓ All losses tested!")
