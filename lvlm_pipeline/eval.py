"""
eval_miou_lvlm.py
使用 LVLM + SAM2 pipeline 对 val 集计算 IoU / mIoU
本地 Windows 运行版本
"""

import os, sys, torch, json, re, cv2, heapq
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

torch.backends.cudnn.enabled = False

# ==================== CONFIG ====================
SAM2_PROJECT_PATH = r"D:\pycharm_projects\NOTRAING\SAM2_LVM\sam2-main"
SAM2_CHECKPOINT   = r"D:\pycharm_projects\NOTRAING\NOTRAING\checkpoints\sam2.1_hiera_base_plus.pt"
SAM2_CONFIG       = "configs/sam2.1/sam2.1_hiera_b+.yaml"
LVLM_MODEL_ID     = "Qwen/Qwen2.5-VL-3B-Instruct"

VAL_IMG_DIR  = r"D:\data\final_dataset\final_dataset\images\val"
VAL_MASK_DIR = r"D:\data\final_dataset\final_dataset\masks\val"
SAVE_DIR     = r"D:\data\final_dataset\eval_results_lvlm"

TOP_K  = 10       # 保存最好的前N张（3类都出现的）
DEVICE = "cuda"   # 改成 "cpu" 如果没有GPU
# ==================== END CONFIG ====================

ORIGINAL_CWD = os.getcwd()

# GT mask 颜色 → 类别（BGR）
GT_COLOR_TO_CLASS = {
    (0,  255,   0): "body",
    (0,    0, 255): "solar_panel",
    (255,  0,   0): "antenna",
}
# 可视化颜色（BGR，与GT一致）
CLASS_COLORS_BGR = {
    "body":        (0,  255,   0),
    "solar_panel": (0,    0, 255),
    "antenna":     (255,  0,   0),
}
CLASS_NAMES = ["body", "solar_panel", "antenna"]

DETECT_CLASSES = {
    # "body":        "卫星的主体结构，最大的中心矩形部件",
    # "solar_panel": "太阳能电池板，位于两侧扁平翼状结构",
    # "antenna":     "天线，细长的杆状突出结构",
    "body": "矩形或细长结构，位于卫星中心，可能有翼状结构",
    "solar_panel": "矩形或细长条形，翼状结构，深色金属光泽，反射光线",
    "antenna": "细长或翼状结构，对称性，深蓝色或黑色，边缘清晰",
}


# ==================== 模型加载 ====================

def load_lvlm():
    print("[1/2] 加载 LVLM (Qwen2.5-VL)...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        LVLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(LVLM_MODEL_ID, local_files_only=True)
    print("✓ LVLM 加载完成\n")
    return model, processor


def load_sam2():
    print("[2/2] 加载 SAM2.1...")
    sys.path.insert(0, SAM2_PROJECT_PATH)
    os.chdir(SAM2_PROJECT_PATH)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    sam2 = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2)
    os.chdir(ORIGINAL_CWD)
    print(f"✓ SAM2.1 加载完成 (device={device})\n")
    return predictor


# ==================== LVLM 推理 ====================

def build_detect_prompt(W, H):
    """含图像尺寸信息，提升坐标准确性"""
    class_desc = "\n".join([f"- {k}: {v}" for k, v in DETECT_CLASSES.items()])
    return f"""你是一个遥感卫星图像目标检测器。图像尺寸为 {W}x{H} 像素（宽x高）。
请检测图像中以下所有类别的目标。

需要检测的类别：
{class_desc}

输出格式（严格JSON列表，不要有任何其他文字）：
[
  {{"class": "body", "bbox_2d": [x1, y1, x2, y2]}},
  {{"class": "solar_panel", "bbox_2d": [x1, y1, x2, y2]}},
  {{"class": "antenna", "bbox_2d": [x1, y1, x2, y2]}}
]

规则：
1. 坐标为像素整数，左上角(x1,y1) 右下角(x2,y2)
2. x 范围 [0, {W}]，y 范围 [0, {H}]
3. 同类别多实例请分别列出
4. 未检测到的类别不要输出
5. 只输出JSON，不要有任何解释"""


def lvlm_infer_single(model, processor, image_path, W, H):
    """单张图像 LVLM 推理"""
    from qwen_vl_utils import process_vision_info

    prompt = build_detect_prompt(W, H)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text",  "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    if "assistant" in out_text:
        out_text = out_text.split("assistant")[-1].strip()
    return out_text


def parse_output(text):
    """解析 LVLM JSON 输出，返回 [{class, bbox_2d}, ...]"""
    for pattern in [r'```json\s*([\s\S]*?)\s*```', r'\[[\s\S]*?\]']:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1) if '```' in pattern else m.group(0))
            except json.JSONDecodeError:
                continue
    return []


# ==================== SAM2 分割 ====================

def sam2_segment(predictor, image_np, prompts):
    """用 bbox 提示 SAM2，返回 {class_name: bool_mask}"""
    os.chdir(SAM2_PROJECT_PATH)
    predictor.set_image(image_np)

    pred_dict = {}
    for p in prompts:
        cls  = p.get("class", "unknown")
        if cls not in CLASS_NAMES:
            continue
        if "bbox_2d" not in p:
            continue

        bbox = np.array(p["bbox_2d"], dtype=np.float32)
        # 边界检查
        H, W = image_np.shape[:2]
        bbox = np.clip(bbox, 0, [W, H, W, H])
        if (bbox[2] - bbox[0]) < 5 or (bbox[3] - bbox[1]) < 5:
            continue

        try:
            masks, scores, _ = predictor.predict(
                box=bbox, multimask_output=False)
            mask = masks[0].astype(bool)
            # 同类别多实例取并集
            if cls in pred_dict:
                pred_dict[cls] = pred_dict[cls] | mask
            else:
                pred_dict[cls] = mask
        except Exception:
            continue

    os.chdir(ORIGINAL_CWD)
    return pred_dict


# ==================== GT mask 解析 ====================

def parse_gt_mask(mask_path):
    """彩色GT mask → {class_name: bool_mask}"""
    mask = cv2.imread(str(mask_path))
    if mask is None:
        return {}
    result = {}
    for (b, g, r), name in GT_COLOR_TO_CLASS.items():
        binary = np.all(mask == np.array([b, g, r]), axis=2)
        if binary.any():
            result[name] = binary
    return result


def has_all_classes(gt_dict):
    return all(n in gt_dict for n in CLASS_NAMES)


# ==================== IoU 计算 ====================

def compute_iou(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / float(union) if union > 0 else 1.0


# ==================== 可视化 ====================

def make_compare_image(image_bgr, gt_mask_path, pred_dict):
    """生成4格对比图: 原图 | GT mask | SAM2 mask | 叠加图"""
    H, W = image_bgr.shape[:2]

    # GT mask
    gt_bgr = cv2.imread(str(gt_mask_path)) if gt_mask_path else np.zeros_like(image_bgr)

    # SAM2 mask（还原为彩色）
    sam2_mask_img = np.zeros_like(image_bgr)
    for cls, mask in pred_dict.items():
        color = CLASS_COLORS_BGR.get(cls, (200, 200, 200))
        sam2_mask_img[mask] = color

    # 叠加图
    overlay = image_bgr.copy()
    for cls, mask in pred_dict.items():
        color = CLASS_COLORS_BGR.get(cls, (200, 200, 200))
        colored = np.zeros_like(overlay)
        colored[mask] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.45, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

    def add_title(img, title):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(out, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return out

    row = np.hstack([
        add_title(image_bgr, "Original"),
        add_title(gt_bgr,        "GT Mask"),
        add_title(sam2_mask_img, "LVLM+SAM2 Mask"),
        add_title(overlay,       "Overlay"),
    ])
    return row


# ==================== 主评估循环 ====================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 加载模型
    lvlm_model, lvlm_processor = load_lvlm()
    sam2_predictor = load_sam2()

    val_images = sorted(Path(VAL_IMG_DIR).glob("*.png"))
    print(f"Val 集共 {len(val_images)} 张图像\n")
    print("="*55)

    iou_per_class  = {n: [] for n in CLASS_NAMES}
    top_k_heap     = []
    skipped        = 0
    all_3cls_count = 0
    error_log      = []

    for img_path in tqdm(val_images, desc="Evaluating"):
        stem      = img_path.stem
        mask_path = Path(VAL_MASK_DIR) / f"{stem}_mask.png"

        if not mask_path.exists():
            skipped += 1
            continue

        try:
            # 读图
            image_bgr = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            H, W = image_bgr.shape[:2]

            # LVLM 检测
            raw_output = lvlm_infer_single(lvlm_model, lvlm_processor, img_path, W, H)
            prompts    = parse_output(raw_output)

            if not prompts:
                skipped += 1
                error_log.append(f"{stem}: LVLM 未输出有效检测")
                continue

            # SAM2 分割
            image_np  = np.array(Image.open(img_path).convert("RGB"))
            pred_dict = sam2_segment(sam2_predictor, image_np, prompts)

            # GT 解析
            gt_dict = parse_gt_mask(mask_path)

        except Exception as e:
            skipped += 1
            error_log.append(f"{stem}: {e}")
            continue

        # 计算 IoU
        ious_this = []
        for name in CLASS_NAMES:
            gt_m   = gt_dict.get(name)
            pred_m = pred_dict.get(name)

            if gt_m is None and pred_m is None:
                continue
            iou = 0.0 if (gt_m is None or pred_m is None) else compute_iou(pred_m, gt_m)

            iou_per_class[name].append(iou)
            ious_this.append(iou)

        if not ious_this:
            continue

        mean_iou = float(np.mean(ious_this))

        # Top-K（仅3类都出现的图）
        if has_all_classes(gt_dict):
            all_3cls_count += 1
            entry = (mean_iou, stem, str(img_path), str(mask_path),
                     image_bgr, gt_dict, pred_dict)
            if len(top_k_heap) < TOP_K:
                heapq.heappush(top_k_heap, entry)
            elif mean_iou > top_k_heap[0][0]:
                heapq.heapreplace(top_k_heap, entry)

    # ===== 输出结果 =====
    print("\n" + "="*55)
    print("【mIoU 评估结果】  LVLM + SAM2")
    print("="*55)
    all_ious = []
    for name in CLASS_NAMES:
        ious = iou_per_class[name]
        if ious:
            mean = np.mean(ious)
            print(f"  {name:15s}: IoU = {mean:.4f}  (n={len(ious)})")
            all_ious.extend(ious)
        else:
            print(f"  {name:15s}: 无样本")

    miou = float(np.mean(all_ious)) if all_ious else 0.0
    print(f"\n  mIoU (全类平均)      : {miou:.4f}")
    print(f"  3类都出现的图像数    : {all_3cls_count}")
    print(f"  跳过/失败图像数      : {skipped}")
    print("="*55)

    # ===== 保存 Top-K 对比图 =====
    print(f"\n保存 Top-{TOP_K} 对比图...")
    top_k_sorted = sorted(top_k_heap, key=lambda x: x[0], reverse=True)

    for rank, entry in enumerate(top_k_sorted, 1):
        mean_iou, stem, img_path, mask_path, image_bgr, gt_dict, pred_dict = entry
        compare = make_compare_image(image_bgr, mask_path, pred_dict)
        save_path = os.path.join(SAVE_DIR, f"top{rank:02d}_iou{mean_iou:.4f}_{stem}.jpg")
        cv2.imwrite(save_path, compare)
        print(f"  Top{rank:02d} | mIoU={mean_iou:.4f} | {stem}")

    # ===== 保存报告 =====
    report_path = os.path.join(SAVE_DIR, "miou_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("LVLM + SAM2 Pipeline — mIoU 评估报告\n")
        f.write(f"LVLM  : {LVLM_MODEL_ID}\n")
        f.write(f"SAM2  : {SAM2_CONFIG}\n")
        f.write(f"Ckpt  : {SAM2_CHECKPOINT}\n")
        f.write("="*55 + "\n")
        for name in CLASS_NAMES:
            ious = iou_per_class[name]
            if ious:
                f.write(f"  {name:15s}: IoU={np.mean(ious):.4f}  (n={len(ious)})\n")
        f.write(f"\n  mIoU              : {miou:.4f}\n")
        f.write(f"  3类全出现图像数   : {all_3cls_count}\n")
        f.write(f"  跳过/失败数       : {skipped}\n\n")
        if error_log:
            f.write("错误日志:\n")
            for e in error_log[:50]:  # 最多记录50条
                f.write(f"  {e}\n")
        f.write("\nTop-K 最优图像（3类全出现）:\n")
        for rank, entry in enumerate(top_k_sorted, 1):
            f.write(f"  Top{rank:02d} | mIoU={entry[0]:.4f} | {entry[1]}\n")

    print(f"\n报告已保存: {report_path}")
    print(f"图像已保存: {SAVE_DIR}")
    print("\n✓ 评估完成！")


if __name__ == "__main__":
    main()