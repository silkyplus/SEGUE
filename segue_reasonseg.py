"""
segue_reasonseg.py — SEGUE 完整推理链路 on ReasonSeg
=====================================================
架构: Qwen2.5-VL (推理) → SAM2 (分割) → 验证 → 迭代修正

流程:
1. Qwen 观察图像 + 推理查询 → 链式思考 + [SEG]几何提示
2. 解析 [SEG] → bbox + keypoints
3. SAM2 → 候选 mask
4. 验证 mask 质量 → 若不合格则迭代修正
5. 输出最终 mask + 推理链

用法 (零样本测试):
    python segue_reasonseg.py --num_samples 10
"""

import sys, os, re, json, copy
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, r'D:\afss')

import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm

# ======================== CONFIG ========================
# LVLM
LVLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# SAM2
SAM2_ROOT = r"D:\afss"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CHECKPOINT = r"D:\pycharm_projects\NOTRAING\NOTRAING\checkpoints\sam2.1_hiera_base_plus.pt"

# SEGUE 参数
MAX_ITERATIONS = 3           # 最多迭代修正 3 次
IOU_IMPROVEMENT_THRESH = 0.02 # IoU 提升低于此值则停止迭代
MAX_IMAGE_SIZE = 1024        # 图像最大边长 (Qwen 处理上限)

# 输出
OUTPUT_DIR = Path(__file__).parent / "reasonseg_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ======================== SEGUE Prompt ========================
SEGUE_SYSTEM_PROMPT = """You are a precise visual reasoning agent for segmentation.

Your task: given an image and a query, reason about WHERE the target is, then output geometric prompts for segmentation.

OUTPUT FORMAT (must follow exactly):
<REASON>
[Your step-by-step reasoning about the target's visual appearance, location, spatial relationship, and any distinguishing features. Be specific about shapes, colors, relative positions.]
</REASON>
<SEG_BOX>
x1 y1 x2 y2
</SEG_BOX>
<SEG_POINTS>
px1 py1
px2 py2
</SEG_POINTS>

Rules:
- <SEG_BOX>: bounding box coordinates in pixels, format "x1 y1 x2 y2" (top-left and bottom-right corners). The box should tightly enclose the target.
- <SEG_POINTS>: two (x, y) points that are definitely INSIDE the target, one near the center and one near a distinctive feature.
- All coordinates are in pixels relative to the image dimensions.
- Be precise with coordinates. Think step by step about the spatial layout before outputting numbers.
"""

SEGUE_VERIFY_PROMPT = """You previously produced this segmentation for the query "{query}":

Previous output:
{prev_output}

Now, examine the following information about the generated mask:
- Mask coverage: {coverage:.1%} of the image
- Mask area: {area} pixels
- Mask centroid: ({cx:.0f}, {cy:.0f})

Does this mask correctly cover the target described in the query? If not, adjust your bounding box and points to improve coverage.

Output the SAME format:
<REASON>
[Your analysis of what was wrong and how to fix it]
</REASON>
<SEG_BOX>
x1 y1 x2 y2
</SEG_BOX>
<SEG_POINTS>
px1 py1
px2 py2
</SEG_POINTS>"""


# ======================== Model Loading ========================
def load_lvlm():
    """加载 Qwen2.5-VL"""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"[LVLM] Loading {LVLM_MODEL_ID}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        LVLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(LVLM_MODEL_ID, local_files_only=True)
    print(f"[LVLM] Ready.")
    return model, processor


def load_sam2(device='cuda'):
    """加载 SAM2"""
    import os as _os
    sam2_dir = SAM2_ROOT + "/sam2"
    orig_cwd = _os.getcwd()
    _os.chdir(sam2_dir)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print(f"[SAM2] Loading from {SAM2_CHECKPOINT}...")
    model = build_sam2(config_file=SAM2_CONFIG, ckpt_path=SAM2_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(model)
    _os.chdir(orig_cwd)
    print(f"[SAM2] Ready.")
    return predictor


# ======================== Prompt Parsing ========================
def parse_segue_output(text):
    """从 Qwen 输出中解析 <REASON>, <SEG_BOX>, <SEG_POINTS>"""
    result = {'reason': '', 'bbox': None, 'points': None, 'raw': text}

    # 抽取 REASON
    reason_match = re.search(r'<REASON>\s*([\s\S]*?)\s*</REASON>', text, re.IGNORECASE)
    if reason_match:
        result['reason'] = reason_match.group(1).strip()

    # 抽取 SEG_BOX
    box_match = re.search(r'<SEG_BOX>\s*([\s\S]*?)\s*</SEG_BOX>', text, re.IGNORECASE)
    if box_match:
        nums = re.findall(r'\d+\.?\d*', box_match.group(1))
        if len(nums) >= 4:
            result['bbox'] = [float(n) for n in nums[:4]]

    # 抽取 SEG_POINTS
    pts_match = re.search(r'<SEG_POINTS>\s*([\s\S]*?)\s*</SEG_POINTS>', text, re.IGNORECASE)
    if pts_match:
        nums = re.findall(r'\d+\.?\d*', pts_match.group(1))
        if len(nums) >= 4:
            result['points'] = [(float(nums[i]), float(nums[i+1])) for i in range(0, len(nums)-1, 2)][:2]

    return result


def lvlm_generate(model, processor, image, prompt_text, max_tokens=512):
    """Qwen2.5-VL 推理"""
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    if "assistant" in out_text:
        out_text = out_text.split("assistant")[-1].strip()
    return out_text


# ======================== Mask Verification ========================
def verify_mask(mask, image_np):
    """自动验证 mask 质量，返回指标字典"""
    if mask is None or mask.sum() == 0:
        return {'valid': False, 'coverage': 0, 'area': 0, 'cx': 0, 'cy': 0, 'edge_density': 0}

    H, W = mask.shape
    area = mask.sum()
    coverage = area / (H * W)

    # 质心
    ys, xs = np.where(mask)
    cx, cy = xs.mean(), ys.mean()

    # 边缘密度 (粗略: 边界附近像素比例)
    from scipy import ndimage
    edges = ndimage.binary_dilation(mask, iterations=2) ^ mask
    edge_density = edges.sum() / max(area, 1)

    return {
        'valid': area > 100,        # 至少 100 像素
        'coverage': coverage,
        'area': int(area),
        'cx': cx, 'cy': cy,
        'edge_density': edge_density,
    }


def compute_mask_iou(mask1, mask2):
    """两个 mask 之间的 IoU"""
    if mask1 is None or mask2 is None:
        return 0.0
    inter = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return inter / max(union, 1)


# ======================== SAM2 Segmentation ========================
def sam2_segment(predictor, image_np, bbox, points):
    """SAM2 分割: bbox + points → mask"""
    predictor.set_image(image_np)

    point_coords = np.array(points, dtype=np.float32) if points else None
    point_labels = np.ones(len(points), dtype=np.int32) if points else None
    box = np.array(bbox, dtype=np.float32) if bbox else None

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=False,
    )
    return masks[0].astype(bool), float(scores[0])


# ======================== SEGUE Main Pipeline ========================
def segue_segment(model, processor, sam2_pred, image_pil, query, image_id=""):
    """
    SEGUE 完整推理链路: 推理 → 分割 → 验证 → 迭代修正
    """
    # Resize 图像到推理尺寸
    w_orig, h_orig = image_pil.size
    scale = min(MAX_IMAGE_SIZE / max(w_orig, h_orig), 1.0)
    w_infer = int(w_orig * scale)
    h_infer = int(h_orig * scale)
    image_infer = image_pil.resize((w_infer, h_infer), Image.BILINEAR)

    # Step 1: Qwen 初始推理
    prompt = SEGUE_SYSTEM_PROMPT + f"\n\nImage size: {w_infer} x {h_infer} pixels\n\nQuery: {query}\n\nNow generate the <REASON>, <SEG_BOX>, and <SEG_POINTS>:"
    output_text = lvlm_generate(model, processor, image_infer, prompt, max_tokens=512)
    parsed = parse_segue_output(output_text)

    if parsed['bbox'] is None:
        return None, {"error": "No bbox parsed", "output": output_text, "iterations": []}

    # 将推理坐标映射回原图
    scale_back = 1.0 / scale
    bbox = [parsed['bbox'][0] * scale_back, parsed['bbox'][1] * scale_back,
            parsed['bbox'][2] * scale_back, parsed['bbox'][3] * scale_back]
    points = [(p[0] * scale_back, p[1] * scale_back) for p in parsed['points']] if parsed['points'] else []

    image_np = np.array(image_pil)

    iterations = [{
        'step': 0,
        'reason': parsed['reason'],
        'bbox': bbox.copy(),
        'points': points.copy() if points else [],
    }]

    # Step 2: SAM2 分割
    mask, score = sam2_segment(sam2_pred, image_np, bbox, points)
    prev_mask = mask
    prev_metrics = verify_mask(mask, image_np)
    iterations[-1]['score'] = score
    iterations[-1]['mask_area'] = prev_metrics['area']

    # Step 3-4: 验证 + 迭代修正
    for it in range(1, MAX_ITERATIONS + 1):
        if not prev_metrics['valid']:
            break

        # 验证: 检查 mask 质量
        metric_text = f"coverage={prev_metrics['coverage']:.3f}, area={prev_metrics['area']}, centroid=({prev_metrics['cx']:.0f},{prev_metrics['cy']:.0f}), edge_density={prev_metrics['edge_density']:.3f}"

        # Qwen 自我修正
        verify_prompt = SEGUE_VERIFY_PROMPT.format(
            query=query,
            prev_output=output_text,
            coverage=prev_metrics['coverage'],
            area=prev_metrics['area'],
            cx=prev_metrics['cx'],
            cy=prev_metrics['cy'],
        ) + f"\n\nCurrent mask metrics: {metric_text}\nImage size: {w_infer} x {h_infer}\n\nProvide improved <SEG_BOX> and <SEG_POINTS>:"

        refine_text = lvlm_generate(model, processor, image_infer, verify_prompt, max_tokens=512)
        refined = parse_segue_output(refine_text)

        if refined['bbox'] is None:
            break  # 无法解析修正 → 停止迭代

        # 映射回原图
        new_bbox = [refined['bbox'][0] * scale_back, refined['bbox'][1] * scale_back,
                    refined['bbox'][2] * scale_back, refined['bbox'][3] * scale_back]
        new_points = [(p[0] * scale_back, p[1] * scale_back) for p in refined['points']] if refined['points'] else []

        # SAM2 重新分割
        new_mask, new_score = sam2_segment(sam2_pred, image_np, new_bbox, new_points)
        new_metrics = verify_mask(new_mask, image_np)

        # 检查是否改进
        iou_with_prev = compute_mask_iou(new_mask, prev_mask)
        improved = new_score > score + IOU_IMPROVEMENT_THRESH

        iterations.append({
            'step': it,
            'reason': refined['reason'],
            'bbox': new_bbox,
            'points': new_points,
            'score': new_score,
            'mask_area': new_metrics['area'],
            'iou_with_prev': iou_with_prev,
            'improved': improved,
        })

        output_text = refine_text
        bbox, points = new_bbox, new_points
        mask, score = new_mask, new_score
        prev_mask = new_mask
        prev_metrics = new_metrics

        if not improved and it >= 2:
            break  # 连续未改进 → 停止

    return mask, {
        'query': query,
        'image_id': image_id,
        'iterations': iterations,
        'final_mask': mask,
        'num_iterations': len(iterations),
    }


# ======================== Evaluation ========================
def compute_giou(pred_mask, gt_mask):
    """Generalized IoU (ReasonSeg 标准指标)"""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / max(union, 1)


def run_evaluation(model, processor, sam2_pred, dataset, num_samples=None, save_vis=True):
    """在 ReasonSeg 上运行评估"""
    results = []
    gious = []

    indices = range(len(dataset))
    if num_samples:
        indices = indices[:num_samples]

    for idx in tqdm(indices, desc="SEGUE"):
        sample = dataset[idx]
        image = sample['image'].convert('RGB')
        query = sample['text']
        gt_mask = np.array(sample['mask'], dtype=bool)
        image_id = sample['image_id']

        mask, info = segue_segment(model, processor, sam2_pred, image, query, image_id)

        if mask is not None:
            # Resize GT mask to match pred mask if needed
            if gt_mask.shape != mask.shape:
                gt_resized = np.array(Image.fromarray(gt_mask).resize(
                    (mask.shape[1], mask.shape[0]), Image.NEAREST))
            else:
                gt_resized = gt_mask

            giou = compute_giou(mask, gt_resized)
            gious.append(giou)
            info['gIoU'] = giou
        else:
            info['gIoU'] = 0.0
            gious.append(0.0)

        results.append(info)

        # 可视化 (每10个保存一张)
        if save_vis and (idx % 10 == 0 or idx < 5):
            save_visualization(image, gt_mask, mask, query, image_id, idx)

    # 汇总
    avg_giou = np.mean(gious) if gious else 0.0
    print(f"\n{'='*60}")
    print(f"  SEGUE Zero-Shot on ReasonSeg (n={len(gious)})")
    print(f"{'='*60}")
    print(f"  Average gIoU:  {avg_giou:.4f}")
    print(f"  > 0.3:         {sum(1 for g in gious if g > 0.3) / len(gious):.1%}")
    print(f"  > 0.5:         {sum(1 for g in gious if g > 0.5) / len(gious):.1%}")
    print(f"  = 0.0:         {sum(1 for g in gious if g == 0) / len(gious):.1%}")

    avg_iters = np.mean([r.get('num_iterations', 1) for r in results])
    print(f"  Avg iterations: {avg_iters:.1f}")

    return results, avg_giou


def save_visualization(image, gt_mask, pred_mask, query, image_id, idx):
    """保存可视化结果"""
    if pred_mask is None:
        return

    w, h = image.size
    if pred_mask.shape != (h, w):
        pred_resized = np.array(Image.fromarray(pred_mask).resize((w, h), Image.NEAREST))
    else:
        pred_resized = pred_mask
    if gt_mask.shape != (h, w):
        gt_resized = np.array(Image.fromarray(gt_mask).resize((w, h), Image.NEAREST))
    else:
        gt_resized = gt_mask

    # 叠加图
    img_np = np.array(image).copy()
    overlay = img_np.copy()

    # GT: 绿色, Pred: 红色
    overlay[gt_resized] = [0, 255, 0]
    overlay[pred_resized] = [255, 0, 0]

    # 混合
    alpha = 0.5
    blended = (img_np * (1 - alpha) + overlay * alpha).astype(np.uint8)
    result = Image.fromarray(blended)

    # Add text
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(result)
    query_short = query[:80] + "..." if len(query) > 80 else query
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    draw.rectangle([0, 0, w, 30], fill=(0, 0, 0, 180))
    draw.text((5, 5), f"GT=green Pred=red | {query_short}", fill=(255, 255, 255), font=font)
    draw.text((5, h-25), f"ID: {image_id}", fill=(255, 255, 255), font=font)

    save_path = OUTPUT_DIR / f"{idx:04d}_{image_id}.jpg"
    result.save(save_path, quality=90)


# ======================== Main ========================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to test')
    parser.add_argument('--save_vis', action='store_true', default=True, help='Save visualizations')
    args = parser.parse_args()

    # Load dataset
    from datasets import load_dataset
    print("Loading ReasonSeg test set...")
    ds = load_dataset('Ricky06662/ReasonSeg_test')['test']
    print(f"Dataset: {len(ds)} samples")

    # Load models
    lvlm_model, lvlm_processor = load_lvlm()
    sam2_pred = load_sam2()

    # Run
    results, avg_giou = run_evaluation(
        lvlm_model, lvlm_processor, sam2_pred, ds,
        num_samples=args.num_samples,
        save_vis=args.save_vis,
    )

    print(f"\nResults saved to: {OUTPUT_DIR}")
    return results


if __name__ == '__main__':
    main()
