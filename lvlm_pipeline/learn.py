"""
auto_prompt_gen.py
Step 1: 从数据集抽样，可视化每个类别的 mask 样例
Step 2: 把样例图喂给 LVLM，让它自己总结每类的视觉特征
Step 3: 把生成的描述自动写入检测 prompt 并保存
"""

import os, sys, torch, cv2, json, random
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ==================== CONFIG ====================
LVLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

TRAIN_IMG_DIR  = r"D:\data\final_dataset\final_dataset\images\val"
TRAIN_MASK_DIR = r"D:\data\final_dataset\final_dataset\masks\val"

# 每个类别抽几张样例给 LVLM 看
SAMPLES_PER_CLASS = 4

# 输出
OUTPUT_DIR        = r"D:\data\final_dataset\prompt_gen"
SAMPLE_VIS_DIR    = os.path.join(OUTPUT_DIR, "class_samples")   # 可视化样例图
GENERATED_PROMPT  = os.path.join(OUTPUT_DIR, "generated_prompt.json")  # 生成的描述
# ==================== END CONFIG ====================

GT_COLOR_TO_CLASS = {
    (0,  255,   0): "body",
    (0,    0, 255): "solar_panel",
    (255,  0,   0): "antenna",
}
CLASS_NAMES = ["body", "solar_panel", "antenna"]
CLASS_COLORS_BGR = {
    "body":        (0,  255,   0),
    "solar_panel": (0,    0, 255),
    "antenna":     (255,  0,   0),
}


# ==================== Step 1: 抽样 + 可视化 ====================

def extract_class_samples():
    """
    遍历数据集，为每个类别收集样例：
    - 原图裁剪到 bbox 区域（让 LVLM 聚焦在目标上）
    - mask 高亮叠加图
    - 并排拼成样例图
    返回：{class_name: [sample_image_path, ...]}
    """
    os.makedirs(SAMPLE_VIS_DIR, exist_ok=True)

    # 收集每类的候选图像
    class_candidates = {n: [] for n in CLASS_NAMES}

    mask_files = sorted(Path(TRAIN_MASK_DIR).glob("*_mask.png"))
    print(f"扫描 {len(mask_files)} 个mask文件...")

    for mask_path in mask_files:
        stem     = mask_path.stem.replace("_mask", "")
        img_path = Path(TRAIN_IMG_DIR) / f"{stem}.png"
        if not img_path.exists():
            continue

        mask_bgr = cv2.imread(str(mask_path))
        for (b, g, r), cls_name in GT_COLOR_TO_CLASS.items():
            binary = np.all(mask_bgr == np.array([b, g, r]), axis=2)
            if binary.any():
                class_candidates[cls_name].append((str(img_path), str(mask_path)))

    # 每类随机抽样
    sample_paths = {}
    for cls_name in CLASS_NAMES:
        candidates = class_candidates[cls_name]
        n = min(SAMPLES_PER_CLASS, len(candidates))
        sampled = random.sample(candidates, n)
        sample_paths[cls_name] = sampled
        print(f"  {cls_name}: 候选 {len(candidates)} 张，抽取 {n} 张")

    # 生成可视化样例图（每类一张拼图）
    sample_vis_paths = {}
    for cls_name, samples in sample_paths.items():
        color = CLASS_COLORS_BGR[cls_name]
        panels = []

        for img_path, mask_path in samples:
            image_bgr = cv2.imread(img_path)
            mask_bgr  = cv2.imread(mask_path)
            H, W = image_bgr.shape[:2]

            # 提取该类 mask
            b, g, r = next(k for k, v in GT_COLOR_TO_CLASS.items() if v == cls_name)
            binary = np.all(mask_bgr == np.array([b, g, r]), axis=2)

            # 找 bbox
            ys, xs = np.where(binary)
            if len(xs) == 0:
                continue
            x1, y1 = max(0, xs.min()-10), max(0, ys.min()-10)
            x2, y2 = min(W, xs.max()+10), min(H, ys.max()+10)

            # 裁剪区域
            crop_orig = image_bgr[y1:y2, x1:x2].copy()
            crop_mask = image_bgr[y1:y2, x1:x2].copy()

            # mask 高亮
            region_mask = binary[y1:y2, x1:x2]
            colored = np.zeros_like(crop_mask)
            colored[region_mask] = color
            crop_mask = cv2.addWeighted(crop_mask, 0.5, colored, 0.5, 0)

            # mask 轮廓
            contours, _ = cv2.findContours(
                region_mask.astype(np.uint8)*255,
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(crop_mask, contours, -1, color, 2)

            # 统一 resize 到 320x240
            target = (320, 240)
            crop_orig = cv2.resize(crop_orig, target)
            crop_mask = cv2.resize(crop_mask, target)

            # 标题
            for img, title in [(crop_orig, "original"), (crop_mask, f"{cls_name} mask")]:
                cv2.rectangle(img, (0,0),(img.shape[1],24),(0,0,0),-1)
                cv2.putText(img, title, (4,17),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

            panels.append(np.hstack([crop_orig, crop_mask]))

        if not panels:
            continue

        # 竖向拼接所有样例
        grid = np.vstack(panels)
        save_path = os.path.join(SAMPLE_VIS_DIR, f"{cls_name}_samples.jpg")
        cv2.imwrite(save_path, grid)
        sample_vis_paths[cls_name] = save_path
        print(f"  ✓ {cls_name} 样例图: {save_path}")

    return sample_vis_paths


# ==================== Step 2: LVLM 自动生成描述 ====================

def load_lvlm():
    print("\n加载 LVLM...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        LVLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(LVLM_MODEL_ID, local_files_only=True)
    print("✓ LVLM 加载完成")
    return model, processor


def lvlm_describe_class(model, processor, cls_name, sample_vis_path):
    """
    给 LVLM 看某类的样例图（原图+mask对），让它总结视觉特征
    """
    from qwen_vl_utils import process_vision_info

    prompt = f"""这张图展示了遥感卫星图像中 "{cls_name}" 类别目标的多个样例。
每行左边是原始图像裁剪，右边是对应的分割mask高亮图（彩色区域就是"{cls_name}"的位置）。

请你仔细观察这些样例，然后用中文描述"{cls_name}"在卫星图像中的视觉特征，要求：
1. 描述它的形状特征（如：矩形、细长、翼状等）
2. 描述它的位置关系（相对于卫星整体在哪里）
3. 描述它的大小特征（占图像比例，长宽比等）
4. 描述任何有助于在图像中定位它的视觉线索

用3-5句话描述，语言要具体，可以直接用于指导目标检测。"""

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": sample_vis_path},
            {"type": "text",  "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    if "assistant" in out_text:
        out_text = out_text.split("assistant")[-1].strip()

    return out_text


def lvlm_generate_final_prompt(model, processor, descriptions, sample_vis_paths):
    """
    把三类描述都给 LVLM，让它生成最终的检测 prompt
    """
    from qwen_vl_utils import process_vision_info

    desc_text = "\n\n".join([
        f"【{cls}】\n{desc}"
        for cls, desc in descriptions.items()
    ])

    # 把三张样例图都传进去
    content = []
    for cls_name in CLASS_NAMES:
        if cls_name in sample_vis_paths:
            content.append({"type": "text",  "text": f"--- {cls_name} 的样例 ---"})
            content.append({"type": "image", "image": sample_vis_paths[cls_name]})

    content.append({"type": "text", "text": f"""
根据上面三类目标的视觉样例和以下描述：

{desc_text}

请生成一段用于目标检测的类别描述，格式如下（直接输出JSON，不要有其他文字）：

{{
  "body": "用于检测body的简洁描述，20字以内",
  "solar_panel": "用于检测solar_panel的简洁描述，20字以内",
  "antenna": "用于检测antenna的简洁描述，20字以内",
  "detection_tips": "检测时的注意事项，尤其是容易混淆或漏检的情况"
}}"""})

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=400, do_sample=False)

    out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    if "assistant" in out_text:
        out_text = out_text.split("assistant")[-1].strip()

    return out_text


# ==================== Step 3: 保存并生成最终 prompt ====================

def save_and_show_results(descriptions, final_prompt_raw):
    import re, json as json_module

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 解析 LVLM 生成的 JSON
    generated = {}
    for pattern in [r'```json\s*([\s\S]*?)\s*```', r'\{[\s\S]*?\}']:
        m = re.search(pattern, final_prompt_raw)
        if m:
            try:
                generated = json_module.loads(
                    m.group(1) if '```' in pattern else m.group(0))
                break
            except Exception:
                continue

    # 保存完整结果
    result = {
        "per_class_descriptions": descriptions,
        "generated_detect_classes": generated,
        "raw_final_output": final_prompt_raw,
    }
    with open(GENERATED_PROMPT, "w", encoding="utf-8") as f:
        json_module.dump(result, f, ensure_ascii=False, indent=2)

    # 打印结果
    print("\n" + "="*60)
    print("【LVLM 自动生成的类别描述】")
    print("="*60)
    for cls, desc in descriptions.items():
        print(f"\n▶ {cls}:")
        print(f"  {desc}")

    print("\n" + "="*60)
    print("【自动生成的 DETECT_CLASSES（可直接用于检测 prompt）】")
    print("="*60)
    if generated:
        for k, v in generated.items():
            print(f"  {k}: {v}")

        # 生成可以直接粘贴到代码里的 Python 字典
        print("\n# ===== 复制以下代码替换你的 DETECT_CLASSES =====")
        print("DETECT_CLASSES = {")
        for cls in CLASS_NAMES:
            if cls in generated:
                print(f'    "{cls}": "{generated[cls]}",')
        print("}")
        if "detection_tips" in generated:
            print(f'\n# 检测提示: {generated["detection_tips"]}')
    else:
        print("  JSON 解析失败，请查看原始输出:")
        print(f"  {final_prompt_raw}")

    print(f"\n完整结果已保存: {GENERATED_PROMPT}")
    print(f"样例图保存在: {SAMPLE_VIS_DIR}")

    return generated


# ==================== 主函数 ====================

def main():
    print("="*60)
    print("  Auto Prompt Generator: LVLM 自动学习类别特征")
    print("="*60)

    # Step 1: 抽样可视化
    print("\n[Step 1] 从数据集抽取样例...")
    sample_vis_paths = extract_class_samples()

    # Step 2: LVLM 学习
    print("\n[Step 2] LVLM 观察样例，生成类别描述...")
    model, processor = load_lvlm()

    descriptions = {}
    for cls_name in CLASS_NAMES:
        if cls_name not in sample_vis_paths:
            print(f"  跳过 {cls_name}（无样例图）")
            continue
        print(f"\n  分析 {cls_name}...")
        desc = lvlm_describe_class(model, processor, cls_name, sample_vis_paths[cls_name])
        descriptions[cls_name] = desc
        print(f"  {cls_name}: {desc[:80]}...")

    # Step 3: 生成最终检测 prompt
    print("\n[Step 3] 生成最终检测 prompt...")
    final_raw = lvlm_generate_final_prompt(model, processor, descriptions, sample_vis_paths)

    # 保存并显示
    generated = save_and_show_results(descriptions, final_raw)

    print("\n✓ 完成！")
    print("\n下一步：把生成的 DETECT_CLASSES 替换到 eval_miou_lvlm_fast.py 里重新评估")


if __name__ == "__main__":
    main()