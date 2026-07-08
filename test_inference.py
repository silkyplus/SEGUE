"""
test_inference.py — SEGUE 推理测试 (无需训练)
==============================================
直接用预训练 GDINO + SAM2 做分割推理并可视化。
"""
import sys, os
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'D:\afss')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from stage1_config import *
from stage1_model import SEGUEStage1

# 输出目录
OUT_DIR = Path(__file__).parent / 'test_output'
OUT_DIR.mkdir(exist_ok=True)


def load_image_for_gdino(img_path, target_size=512):
    """加载图像并转换为 GDINO 输入格式"""
    import torchvision.transforms.functional as TF

    img = Image.open(img_path).convert('RGB')
    orig_size = img.size  # (W, H)

    # Resize 保持宽高比，长边 = target_size
    img_resized = img.copy()
    w, h = img.size
    if max(w, h) > target_size:
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        # 确保是 16 的倍数
        new_w = (new_w // 16) * 16
        new_h = (new_h // 16) * 16
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # To tensor + normalize
    img_np = np.array(img_resized)
    img_t = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    img_t = TF.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    return img_t, img_resized, orig_size


def visualize_result(image_pil, boxes_cxcywh, masks, scores, phrases, save_path):
    """可视化检测框 + 分割 mask + 关键点"""
    img = image_pil.copy().convert('RGBA')
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # 颜色表
    colors = ['#FF4444', '#44FF44', '#4488FF', '#FFAA00', '#FF44FF',
              '#44FFFF', '#AAFF44', '#FF8844']

    for i, (box, mask, score, phrase) in enumerate(zip(boxes_cxcywh, masks, scores, phrases)):
        color = colors[i % len(colors)]
        color_rgb = tuple(int(color[j+1:j+3], 16) for j in (0, 2, 4))
        color_rgba = color_rgb + (100,)

        # 画 mask 叠加
        if mask is not None and mask.sum() > 0:
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask_np, mode='L').resize((w, h), Image.NEAREST)
            overlay = Image.new('RGBA', (w, h), color_rgba)
            img = Image.composite(overlay, img, mask_pil)

        # 画 bbox
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 标签
        label = f"{phrase} ({score:.2f})"
        draw.rectangle([x1, max(0, y1-22), x1 + len(label)*8, y1], fill=color)
        draw.text((x1+2, max(0, y1-20)), label, fill='white')

    # 保存
    img_rgb = img.convert('RGB')
    img_rgb.save(save_path, quality=95)
    print(f"  Saved: {save_path}")

    # 同时保存单独的 mask 图
    mask_only = np.zeros((h, w), dtype=np.uint8)
    for i, (box, mask, _, _) in enumerate(zip(boxes_cxcywh, masks, scores, phrases)):
        if mask is not None and mask.sum() > 0:
            mask_np = mask.cpu().numpy().astype(bool)
            # resize to original size
            mask_resized = Image.fromarray(mask_np).resize((w, h), Image.NEAREST)
            mask_arr = np.array(mask_resized)
            mask_only[mask_arr] = (i + 1) * 40  # 不同灰度表示不同目标

    mask_path = save_path.with_suffix('.mask.png')
    Image.fromarray(mask_only).save(mask_path)
    print(f"  Mask:  {mask_path}")


def main():
    print("=" * 60)
    print("  SEGUE Inference Test (no training)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Load Model ----
    print("\nLoading SEGUE model...")
    model = SEGUEStage1(device=device)
    model.eval()
    print("Model loaded!")

    # ---- Find test images ----
    val_img_dir = Path(VAL_IMG_DIR)
    val_images = sorted(list(val_img_dir.glob('*.png')) + list(val_img_dir.glob('*.jpg')))
    if len(val_images) == 0:
        print(f"No images found in {val_img_dir}")
        return
    print(f"\nFound {len(val_images)} validation images")

    # ---- Test on first 5 images ----
    test_imgs = val_images[:5]
    print(f"Testing on {len(test_imgs)} images...\n")

    for img_idx, img_path in enumerate(test_imgs):
        print(f"[{img_idx+1}/{len(test_imgs)}] {img_path.name}")

        # 加载图像
        img_t, img_pil, orig_size = load_image_for_gdino(img_path)
        img_t = img_t.unsqueeze(0).to(device)  # [1, 3, H, W]

        # 推理 (使用所有类别作为查询)
        caption = TEXT_QUERY  # "body. solar_panel. antenna."
        print(f"  Query: '{caption}'")

        with torch.no_grad():
            outputs = model(img_t, [caption], run_sam2=True)

        # 解析输出
        gdino_boxes = outputs['gdino_bboxes'][0].cpu()
        gdino_scores = outputs['gdino_scores'][0].cpu()
        gdino_phrases = outputs['gdino_phrases'][0]
        pred_boxes = outputs['pred_bboxes'][0].cpu()
        pred_masks = outputs['pred_masks'][0].cpu()

        n_det = len(gdino_boxes)
        print(f"  GDINO detections: {n_det}")

        if n_det > 0:
            for i in range(min(n_det, 5)):
                print(f"    [{i}] {gdino_phrases[i]}: score={gdino_scores[i]:.3f}, "
                      f"box=[{gdino_boxes[i][0]:.2f},{gdino_boxes[i][1]:.2f},"
                      f"{gdino_boxes[i][2]:.2f},{gdino_boxes[i][3]:.2f}]")

            # 可视化
            save_path = OUT_DIR / f"{img_path.stem}_result.jpg"
            visualize_result(
                img_pil,
                pred_boxes if len(pred_boxes) > 0 else gdino_boxes,
                pred_masks if len(pred_masks) > 0 else [None] * n_det,
                gdino_scores,
                gdino_phrases,
                save_path,
            )
        else:
            print("    (no detections above threshold)")
            # 尝试用更低阈值
            print("    Trying with GDINO only (lower threshold)...")
            # 直接调用 GDINO
            from grounding_dino.groundingdino.util.inference import predict as gdino_predict
            from grounding_dino.groundingdino.util.inference import load_image as gdino_load

            # 用 GDINO 的原始方式再试一次
            img_np_raw, img_t_raw = gdino_load(str(img_path))
            boxes_raw, logits_raw, phrases_raw = gdino_predict(
                model.gdino, img_t_raw, caption,
                box_threshold=0.2, text_threshold=0.15, device=device
            )
            print(f"    GDINO raw detections (thresh=0.2): {len(boxes_raw)}")
            for i, (b, s, p) in enumerate(zip(boxes_raw, logits_raw, phrases_raw)):
                print(f"      [{i}] {p}: score={s:.3f}, box={b.tolist()}")

    print(f"\nDone! Results saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
