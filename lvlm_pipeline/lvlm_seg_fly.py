"""
Think2Seg-RS 推理管道：LVLM + SAM2.1 解耦分割
支持两种模式：
  MODE = "detect"   → LVLM 检测固定类别（替代YOLO）
  MODE = "reason"   → LVLM 理解自然语言查询（原始功能）
"""

import os, sys, torch, json, re, cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# ==================== CONFIG ====================
SAM2_PROJECT_PATH = r"D:\pycharm_projects\NOTRAING\SAM2_LVM\sam2-main"
SAM2_CHECKPOINT   = r"D:\pycharm_projects\NOTRAING\NOTRAING\checkpoints\sam2.1_hiera_base_plus.pt"
SAM2_CONFIG       = "configs/sam2.1/sam2.1_hiera_b+.yaml"
LVLM_MODEL_ID     = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH        = r"D:\data\final_dataset\final_dataset\images\val\img_resize_518.png"
OUTPUT_DIR        = r"D:\pycharm_projects\NOTRAING\SAM2_LVM\output_pipeline"

# ===== 切换模式 =====
MODE  = "detect"          # "detect" 检测固定类别 | "reason" 自然语言查询
QUERY = "卫星的太阳能电池板"  # 仅 MODE="reason" 时生效

# detect 模式下要检测的类别及描述
DETECT_CLASSES = {
    "body":        "卫星的主体结构，最大的中心部件",
    "solar_panel": "太阳能电池板，扁平的翼状结构",
    "antenna":     "天线，细长的杆状突出结构",
}

# GT mask 颜色（与你原有pipeline保持一致，BGR）
CLASS_COLORS_BGR = {
    "body":        (0, 255,   0),
    "solar_panel": (0,   0, 255),
    "antenna":     (255,  0,   0),
}
# ==================== END CONFIG ====================

ORIGINAL_CWD = os.getcwd()


# ========== Prompt 构建 ==========

def build_detect_prompt():
    """多类别检测 prompt（替代YOLO）"""
    class_desc = "\n".join(
        [f"- {k}: {v}" for k, v in DETECT_CLASSES.items()]
    )
    return f"""你是一个遥感卫星图像目标检测器。
请检测图像中以下所有类别的目标。

需要检测的类别：
{class_desc}

输出格式（严格JSON列表，不要有其他文字）：
[
  {{"class": "body", "bbox_2d": [x1, y1, x2, y2]}},
  {{"class": "solar_panel", "bbox_2d": [x1, y1, x2, y2]}},
  {{"class": "antenna", "bbox_2d": [x1, y1, x2, y2]}}
]

规则：
1. 坐标为像素整数，左上角(x1,y1) 右下角(x2,y2)
2. 同类别多实例请分别列出
3. 未检测到的类别不要输出
4. 只输出JSON，不要有任何解释文字"""


def build_reason_prompt(query):
    """自然语言推理 prompt（原始功能）"""
    return f"""你是一个视觉提示生成器。根据图像和描述，为分割任务生成几何提示。

目标描述: {query}

输出格式（严格JSON列表，不要有其他文字）：
[
  {{
    "class": "target",
    "bbox_2d": [x1, y1, x2, y2],
    "positive_points": [[px1, py1], [px2, py2]]
  }}
]

规则：
1. 坐标为像素整数
2. positive_points 选在目标内部有代表性的位置
3. 未找到目标输出 []
4. 只输出JSON"""


# ========== 模型加载 ==========

def load_lvlm():
    print("\n[Step 1] 加载 LVLM (Qwen2.5-VL)...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        LVLM_MODEL_ID, torch_dtype=torch.float16,
        device_map="auto", attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(LVLM_MODEL_ID, local_files_only=True)
    print("✓ LVLM 加载完成")
    return model, processor


def load_sam2():
    print("\n[Step 2] 加载 SAM2.1...")
    sys.path.insert(0, SAM2_PROJECT_PATH)
    os.chdir(SAM2_PROJECT_PATH)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2 = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2)
    os.chdir(ORIGINAL_CWD)
    print(f"✓ SAM2.1 加载完成 (device={device})")
    return predictor


# ========== LVLM 推理 ==========

def lvlm_infer(model, processor, image_path):
    print(f"\n[Step 3] LVLM 推理 (MODE={MODE})...")
    from qwen_vl_utils import process_vision_info

    prompt = build_detect_prompt() if MODE == "detect" else build_reason_prompt(QUERY)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    print(f"LVLM 输出:\n{'-'*40}\n{out_text}\n{'-'*40}")
    return out_text


def parse_output(text):
    """解析 LVLM JSON 输出"""
    for pattern in [r'```json\s*([\s\S]*?)\s*```', r'\[[\s\S]*?\]']:
        m = re.search(pattern, text)
        if m:
            try:
                result = json.loads(m.group(1) if '```' in pattern else m.group(0))
                print(f"✓ JSON 解析成功，检测到 {len(result)} 个目标")
                for item in result:
                    print(f"  class={item.get('class')}  bbox={item.get('bbox_2d')}")
                return result
            except json.JSONDecodeError:
                continue
    print("✗ JSON 解析失败")
    return []


# ========== SAM2 分割 ==========

def sam2_segment(predictor, image_np, prompts):
    print(f"\n[Step 4] SAM2 分割 ({len(prompts)} 个目标)...")
    os.chdir(SAM2_PROJECT_PATH)
    predictor.set_image(image_np)

    results = []
    for i, p in enumerate(prompts):
        cls_name = p.get("class", f"target_{i}")
        bbox = np.array(p["bbox_2d"], dtype=np.float32) if "bbox_2d" in p else None

        # detect 模式只用 bbox；reason 模式 bbox + points
        point_coords, point_labels = None, None
        if MODE == "reason" and "positive_points" in p:
            pts = p["positive_points"]
            if pts:
                point_coords = np.array(pts, dtype=np.float32)
                point_labels = np.ones(len(pts), dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=bbox,
            multimask_output=False,
        )
        mask = masks[0].astype(bool)
        print(f"  [{cls_name}] score={scores[0]:.4f}  pixels={mask.sum()}")
        results.append({"class": cls_name, "mask": mask, "score": scores[0], "bbox": bbox})

    os.chdir(ORIGINAL_CWD)
    return results


# ========== 可视化 ==========

def visualize(image_path, seg_results, output_dir):
    print(f"\n[Step 5] 保存结果到 {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    image_bgr = cv2.imread(image_path)
    overlay   = image_bgr.copy()

    for r in seg_results:
        cls   = r["class"]
        mask  = r["mask"]
        score = r["score"]
        bbox  = r["bbox"]
        color = CLASS_COLORS_BGR.get(cls, (200, 200, 200))

        # 半透明 mask
        colored = np.zeros_like(overlay)
        colored[mask] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.45, 0)

        # 轮廓
        contours, _ = cv2.findContours(
            mask.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        # bbox
        if bbox is not None:
            x1,y1,x2,y2 = map(int, bbox)
            cv2.rectangle(overlay, (x1,y1),(x2,y2), color, 2)
            label = f"{cls} {score:.2f}"
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(overlay,(x1,y1-th-8),(x1+tw+4,y1),color,-1)
            cv2.putText(overlay, label,(x1+2,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)

    # 并排对比图
    combined = np.hstack([image_bgr, overlay])
    save_path = os.path.join(output_dir, f"result_MODE_{MODE}.jpg")
    cv2.imwrite(save_path, combined)
    print(f"✓ 已保存: {save_path}")


# ========== 主函数 ==========

def main():
    print("="*60)
    print(f"  Think2Seg-RS Pipeline  MODE={MODE}")
    print("="*60)

    lvlm_model, lvlm_processor = load_lvlm()
    sam2_predictor = load_sam2()

    raw_output = lvlm_infer(lvlm_model, lvlm_processor, IMAGE_PATH)
    prompts    = parse_output(raw_output)

    if not prompts:
        print("未检测到目标，退出")
        return

    image_np  = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    seg_results = sam2_segment(sam2_predictor, image_np, prompts)

    visualize(IMAGE_PATH, seg_results, OUTPUT_DIR)
    print("\n✓ 全部完成！")


if __name__ == "__main__":
    main()