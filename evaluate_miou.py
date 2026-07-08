"""
evaluate_miou.py — SEGUE mIoU 评估
==================================
1. 先检查 pred/GT mask 格式（一张图）
2. 在整个验证集上运行 mIoU 评估
"""
import sys, os
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'D:\afss')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from stage1_config import *
from stage1_model import SEGUEStage1

# ==============================================================
# 评估用类别定义 (与 DINOv3seg/eval.py 一致: 4 类含 background)
# ==============================================================
EVAL_CLASS_NAMES = ['background', 'body', 'solar_panel', 'antenna']
EVAL_NUM_CLASSES = len(EVAL_CLASS_NAMES)

# GT mask RGB → class index (0=bg, 1=body, 2=solar_panel, 3=antenna)
EVAL_RGB_TO_CLASS = {
    (0, 0, 0):     0,   # background
    (0, 255, 0):   1,   # body (绿色)
    (255, 0, 0):   2,   # solar_panel (红色)
    (0, 0, 255):   3,   # antenna (蓝色)
}

# ---- GT mask RGB → class index ----
def rgb_to_class(mask_rgb):
    """(H,W,3) uint8 → (H,W) int64 class index (0=bg, 1=body, 2=solar, 3=antenna)"""
    H, W, _ = mask_rgb.shape
    out = np.zeros((H, W), dtype=np.int64)  # default=0 (background)
    for rgb, cls in EVAL_RGB_TO_CLASS.items():
        if cls == 0:
            continue  # background already default
        r, g, b = rgb
        match = (mask_rgb[..., 0] == r) & (mask_rgb[..., 1] == g) & (mask_rgb[..., 2] == b)
        out[match] = cls
    return out


def phrase_to_class_idx(phrase):
    """GDINO 检测短语 → 类别索引 (0=bg, 1=body, 2=solar_panel, 3=antenna)

    处理 GDINO 不完美的短语输出:
    - 'solar _ panel', 'solar panel', 'panel' → solar_panel
    - 'body', 'body antenna' → body (优先)
    - 'antenna' → antenna
    """
    phrase_lower = phrase.lower().replace(' ', '').replace('_', '')

    # 优先匹配更具体的
    if 'solar' in phrase_lower:
        return 2  # solar_panel
    if 'antenna' in phrase_lower:
        return 3  # antenna
    if 'body' in phrase_lower:
        return 1  # body
    if 'panel' in phrase_lower:
        return 2  # solar_panel (panel alone)

    return 0  # unknown → background


def pred_masks_to_class_map(pred_masks, phrases, image_size):
    """将 per-object masks 合并为 class-index map"""
    H, W = image_size
    class_map = np.zeros((H, W), dtype=np.int64)

    if len(pred_masks) == 0:
        return class_map

    for i, (mask, phrase) in enumerate(zip(pred_masks, phrases)):
        cls_idx = phrase_to_class_idx(phrase)
        if cls_idx == 0:
            continue  # skip unknown

        # mask → bool, resize to image size
        mask_np = mask.cpu().numpy()
        if mask_np.shape != (H, W):
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((W, H), Image.NEAREST)
            mask_np = (np.array(mask_pil) > 127).astype(bool)
        else:
            mask_np = mask_np > 0.5

        # 后处理的 mask 覆盖前面的（按分数排序更好，暂按顺序）
        class_map[mask_np] = cls_idx

    return class_map


def compute_confusion_matrix(pred_cls, gt_cls, num_classes=EVAL_NUM_CLASSES):
    """与 eval.py 一致: 用 bincount 构建混淆矩阵"""
    # 注意: EVAL_NUM_CLASSES=4 for satellite, CLASS_NAMES=['background','body','solar_panel','antenna']
    valid = (gt_cls >= 0) & (gt_cls < num_classes)
    k = gt_cls[valid].astype(np.int64) * num_classes + pred_cls[valid].astype(np.int64)
    binc = np.bincount(k, minlength=num_classes * num_classes)
    return binc.reshape(num_classes, num_classes)


def iou_from_confusion(conf):
    """从混淆矩阵计算 per-class IoU"""
    tp = np.diag(conf).astype(np.float64)
    gt_count = conf.sum(axis=1).astype(np.float64)
    pd_count = conf.sum(axis=0).astype(np.float64)
    union = gt_count + pd_count - tp
    iou = np.where(union > 0, tp / union, np.nan)
    return iou


# ====================================================================
# Main
# ====================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ---- Load Model ----
    print("=" * 60)
    print("  Loading SEGUE model...")
    print("=" * 60)
    model = SEGUEStage1(device=device)
    model.eval()
    print("Model loaded.\n")

    # ---- PHASE 1: Check mask format (1 sample) ----
    print("=" * 60)
    print("  PHASE 1: Mask Format Check")
    print("=" * 60)

    val_img_dir = Path(VAL_IMG_DIR)
    val_mask_dir = Path(VAL_MASK_DIR)
    val_imgs = sorted(list(val_img_dir.glob('*.png')))

    test_img_path = val_imgs[0]
    test_mask_path = val_mask_dir / f"{test_img_path.stem}_mask.png"

    print(f"Image: {test_img_path}")
    print(f"GT Mask: {test_mask_path}")

    # Load GT
    gt_rgb = np.array(Image.open(test_mask_path).convert('RGB'))
    gt_cls = rgb_to_class(gt_rgb)
    print(f"\nGT mask shape: {gt_rgb.shape}, dtype: {gt_rgb.dtype}")
    print(f"GT RGB unique colors: {np.unique(gt_rgb.reshape(-1, 3), axis=0)}")
    print(f"GT class indices: {np.unique(gt_cls)}")
    for cls_idx in np.unique(gt_cls):
        name = EVAL_CLASS_NAMES[cls_idx] if cls_idx < len(EVAL_CLASS_NAMES) else '?'
        count = (gt_cls == cls_idx).sum()
        print(f"  class {cls_idx} ({name}): {count} px ({100*count/gt_cls.size:.1f}%)")

    # Load Image
    import torchvision.transforms.functional as TF
    img_pil = Image.open(test_img_path).convert('RGB')
    orig_w, orig_h = img_pil.size

    # Resize to fit GDINO (长边 512, 保持 16 的倍数)
    scale = 512 / max(orig_w, orig_h)
    new_w = (int(orig_w * scale) // 16) * 16
    new_h = (int(orig_h * scale) // 16) * 16
    img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)

    img_np = np.array(img_resized)
    img_t = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    img_t = TF.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    img_t = img_t.unsqueeze(0).to(device)

    print(f"\nImage: original={orig_w}x{orig_h}, model_input={new_w}x{new_h}, tensor={img_t.shape}")

    # Inference
    print(f"\nRunning inference...")
    with torch.no_grad():
        outputs = model(img_t, [TEXT_QUERY], run_sam2=True)

    gdino_boxes = outputs['gdino_bboxes'][0].cpu()
    gdino_scores = outputs['gdino_scores'][0].cpu()
    gdino_phrases = outputs['gdino_phrases'][0]
    pred_masks = outputs['pred_masks'][0].cpu()

    print(f"GDINO detections: {len(gdino_boxes)}")
    for i in range(len(gdino_boxes)):
        cls_idx = phrase_to_class_idx(gdino_phrases[i])
        print(f"  [{i}] phrase='{gdino_phrases[i]}' → class={cls_idx} ({EVAL_CLASS_NAMES[cls_idx]}), "
              f"score={gdino_scores[i]:.3f}")

    # Pred masks
    print(f"\nPred masks: {pred_masks.shape if len(pred_masks) > 0 else '(empty)'}")
    if len(pred_masks) > 0:
        print(f"  dtype: {pred_masks.dtype}, min={pred_masks.min():.4f}, max={pred_masks.max():.4f}")
        print(f"  per-mask foreground pixels:")
        for i in range(len(pred_masks)):
            fg = (pred_masks[i] > 0.5).sum().item() if pred_masks[i].numel() > 0 else 0
            print(f"    [{i}]: {fg} px")

    # Merge to class map
    pred_cls = pred_masks_to_class_map(pred_masks, gdino_phrases, (new_h, new_w))
    print(f"\nPred class map: shape={pred_cls.shape}, unique classes={np.unique(pred_cls)}")
    for cls_idx in np.unique(pred_cls):
        name = EVAL_CLASS_NAMES[cls_idx] if cls_idx < len(EVAL_CLASS_NAMES) else '?'
        count = (pred_cls == cls_idx).sum()
        print(f"  class {cls_idx} ({name}): {count} px ({100*count/pred_cls.size:.1f}%)")

    # Resize GT to match pred size for comparison
    gt_cls_resized = np.array(Image.fromarray(gt_cls.astype(np.uint8)).resize(
        (new_w, new_h), Image.NEAREST))
    print(f"\nGT class map (resized to {new_w}x{new_h}): unique classes={np.unique(gt_cls_resized)}")

    # Quick IoU
    conf = compute_confusion_matrix(pred_cls, gt_cls_resized)
    iou = iou_from_confusion(conf)
    print(f"\nPer-class IoU (single image):")
    for i, name in enumerate(EVAL_CLASS_NAMES):
        print(f"  {name:15s}: {iou[i]:.4f}" if not np.isnan(iou[i]) else f"  {name:15s}: N/A (no GT)")
    valid_iou = [v for v in iou if not np.isnan(v)]
    print(f"  mIoU (valid): {np.mean(valid_iou):.4f}")

    # ---- PHASE 2: Full validation ----
    print("\n\n" + "=" * 60)
    print("  PHASE 2: Full Validation mIoU")
    print("=" * 60)
    print(f"Evaluating on {len(val_imgs)} images...")

    full_conf = np.zeros((EVAL_NUM_CLASSES, EVAL_NUM_CLASSES), dtype=np.int64)
    n_processed = 0
    no_det = 0

    for idx, img_path in enumerate(val_imgs):
        mask_path = val_mask_dir / f"{img_path.stem}_mask.png"
        if not mask_path.exists():
            continue

        # GT
        gt_rgb = np.array(Image.open(mask_path).convert('RGB'))
        gt_cls = rgb_to_class(gt_rgb)

        # Image
        img_pil = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size
        scale = 512 / max(orig_w, orig_h)
        new_w = max(16, (int(orig_w * scale) // 16) * 16)
        new_h = max(16, (int(orig_h * scale) // 16) * 16)
        img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)

        img_np = np.array(img_resized)
        img_t = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
        img_t = TF.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        img_t = img_t.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(img_t, [TEXT_QUERY], run_sam2=True)

        gdino_phrases = outputs['gdino_phrases'][0]
        pred_masks = outputs['pred_masks'][0].cpu()

        if len(pred_masks) == 0:
            no_det += 1

        # Merge to class map
        pred_cls = pred_masks_to_class_map(pred_masks, gdino_phrases, (new_h, new_w))

        # Resize GT to match
        gt_cls_resized = np.array(Image.fromarray(gt_cls.astype(np.uint8)).resize(
            (new_w, new_h), Image.NEAREST))

        full_conf += compute_confusion_matrix(pred_cls, gt_cls_resized)
        n_processed += 1

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(val_imgs)}] processed...")

    # ---- Results ----
    print(f"\nProcessed: {n_processed} images ({no_det} with no detections)")
    iou = iou_from_confusion(full_conf)

    print("\n" + "=" * 60)
    print("  SEGUE mIoU Results (Dataset-wise)")
    print("=" * 60)
    fg_ious = []
    for i, name in enumerate(EVAL_CLASS_NAMES):
        val = iou[i]
        if np.isnan(val):
            print(f"  {name:15s}: N/A")
        else:
            print(f"  {name:15s}: {val:.4f}")
            if name != 'background':
                fg_ious.append(val)

    miou_fg = np.mean(fg_ious) if fg_ious else 0.0
    miou_all = np.mean([v for v in iou if not np.isnan(v)])
    print(f"  {'─' * 20}")
    print(f"  mIoU (fg 3 classes):  {miou_fg:.4f}")
    print(f"  mIoU (all valid):     {miou_all:.4f}")

    # Confusion matrix
    print(f"\nConfusion Matrix (rows=GT, cols=Pred):")
    header = "         " + " ".join(f"{EVAL_CLASS_NAMES[i]:>8s}" for i in range(len(EVAL_CLASS_NAMES)))
    print(header)
    for i, name in enumerate(EVAL_CLASS_NAMES):
        row = " ".join(f"{full_conf[i, j]:8d}" for j in range(len(EVAL_CLASS_NAMES)))
        print(f"  {name:>7s}: {row}")

    print("\nDone!")


if __name__ == '__main__':
    main()
