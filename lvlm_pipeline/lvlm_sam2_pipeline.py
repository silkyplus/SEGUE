"""
Think2Seg-RS 推理管道：LVLM + SAM2.1 解耦分割
=============================================
基于论文 "Bridging Semantics and Geometry: A Decoupled LVLM–SAM Framework 
for Reasoning Segmentation in Remote Sensing"

流程：
1. LVLM (Qwen2.5-VL) 分析图像 + 查询 → 输出 JSON (bbox + positive_points)
2. 解析 JSON 得到几何提示
3. SAM2.1 接收几何提示 → 生成分割掩码
4. 可视化并保存结果

使用方法：
1. 修改 CONFIG 部分的路径和查询
2. 运行: python lvlm_sam2_pipeline.py
"""

import os
import sys
import torch
import json
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==================== CONFIG ====================
# SAM2 项目路径
SAM2_PROJECT_PATH = r"D:\pycharm_projects\NOTRAING\SAM2_LVM\sam2-main"

# SAM2.1 权重和配置
SAM2_CHECKPOINT = r"D:\pycharm_projects\NOTRAING\NOTRAING\checkpoints\sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# LVLM 模型
LVLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# 输入图像路径
IMAGE_PATH = r"D:\data\final_dataset\final_dataset\images\val\img_resize_518.png"

# 输出目录
OUTPUT_DIR = r"D:\pycharm_projects\NOTRAING\SAM2_LVM\output_pipeline"

# ============ 修改这里：指定要分割的目标 ============
# QUERY = "如果我要切菜，作为人，我的手需要抓的厨具的部分，而不是整个厨具"
QUERY = "天空飞行器的身体"

# QUERY = "如果我要切菜，作为菜，被切的厨具的部分，而不是整个厨具"
# QUERY = "切菜的时候，需要把菜放在的部分"


# ==================================================

# ==================== END CONFIG ====================

# 保存原始工作目录
ORIGINAL_CWD = os.getcwd()


def load_lvlm():
    """加载 LVLM 模型 (Qwen2.5-VL)"""
    print("\n" + "=" * 60)
    print("  Step 1: 加载 LVLM (Qwen2.5-VL)")
    print("=" * 60)
    
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    
    print(f"模型: {LVLM_MODEL_ID}")
    print("正在加载...")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        LVLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    
    processor = AutoProcessor.from_pretrained(LVLM_MODEL_ID, local_files_only=True)
    
    print("✓ LVLM 加载完成！")
    return model, processor


def load_sam2():
    """加载 SAM2.1 模型"""
    print("\n" + "=" * 60)
    print("  Step 2: 加载 SAM2.1")
    print("=" * 60)
    
    # 添加 SAM2 到路径并切换工作目录
    sys.path.insert(0, SAM2_PROJECT_PATH)
    os.chdir(SAM2_PROJECT_PATH)
    
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"配置: {SAM2_CONFIG}")
    print("正在加载...")
    
    sam2 = build_sam2(
        config_file=SAM2_CONFIG,
        ckpt_path=SAM2_CHECKPOINT,
        device=device,
    )
    predictor = SAM2ImagePredictor(sam2)
    
    # 切回原始目录
    os.chdir(ORIGINAL_CWD)
    
    print("✓ SAM2.1 加载完成！")
    return predictor


def lvlm_generate_prompts(model, processor, image_path, query):
    """
    使用 LVLM 生成几何提示
    
    输入: 图像路径, 自然语言查询
    输出: JSON 格式的 bbox 和 positive_points
    """
    print("\n" + "=" * 60)
    print("  Step 3: LVLM 推理 - 生成几何提示")
    print("=" * 60)
    print(f"查询: {query}")
    
    from qwen_vl_utils import process_vision_info
    
    # 构建提示词（参考 Think2Seg-RS 论文的 Instruction Template）
    prompt = f"""你是一个视觉提示生成器。请根据图像和问题，为分割任务生成几何提示。

任务要求：
1. 找到图像中符合描述的目标
2. 为每个目标提供：
   - bbox_2d: [x1, y1, x2, y2] 目标的边界框（像素坐标，左上角和右下角）
   - positive_points: [[px1, py1], [px2, py2]] 目标内部的两个点（像素坐标）
3. 所有坐标必须是整数
4. 只输出 JSON 列表，不要有其他文字
5. 如果没有找到目标，输出空列表 []

目标描述: {query}

请直接输出JSON:"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    print("正在推理...")
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    
    # 提取 assistant 回复部分
    if "assistant" in out_text:
        out_text = out_text.split("assistant")[-1].strip()
    
    print("\nLVLM 输出:")
    print("-" * 40)
    print(out_text)
    print("-" * 40)
    
    return out_text


def parse_json_output(text):
    """从 LVLM 输出中解析 JSON"""
    # 方式1：查找 ```json ... ``` 块
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 方式2：查找 [ ... ] 数组
        json_match = re.search(r'\[[\s\S]*?\]', text)
        if json_match:
            json_str = json_match.group(0)
        else:
            # 方式3：整个文本可能就是 JSON
            json_str = text.strip()
    
    try:
        json_str = json_str.strip()
        result = json.loads(json_str)
        print(f"✓ JSON 解析成功，检测到 {len(result)} 个目标")
        return result
    except json.JSONDecodeError as e:
        print(f"✗ JSON 解析失败: {e}")
        return []


def sam2_segment(predictor, image_np, prompts):
    """
    使用 SAM2.1 进行分割
    
    输入: SAM2 预测器, 图像数组, 几何提示列表
    输出: 合并后的掩码, 各个单独的掩码列表
    """
    print("\n" + "=" * 60)
    print("  Step 4: SAM2.1 分割执行")
    print("=" * 60)
    print(f"目标数量: {len(prompts)}")
    
    # 切换到 SAM2 目录
    os.chdir(SAM2_PROJECT_PATH)
    
    # 设置图像
    predictor.set_image(image_np)
    
    all_masks = []
    all_scores = []
    
    for i, prompt_data in enumerate(prompts):
        print(f"\n处理目标 {i + 1}:")
        
        # 提取 bbox
        bbox = None
        if "bbox_2d" in prompt_data:
            bbox = np.array(prompt_data["bbox_2d"], dtype=np.float32)
            print(f"  Bbox: {bbox.tolist()}")
        
        # 提取 positive points
        point_coords = None
        point_labels = None
        if "positive_points" in prompt_data:
            points = prompt_data["positive_points"]
            if len(points) > 0:
                point_coords = np.array(points, dtype=np.float32)
                point_labels = np.ones(len(points), dtype=np.int32)
                print(f"  Points: {points}")
        
        # SAM2 预测
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=bbox,
            multimask_output=False,
        )
        
        mask = masks[0].astype(bool)  # 确保是布尔类型
        score = scores[0]
        print(f"  置信度: {score:.4f}")
        
        all_masks.append(mask)
        all_scores.append(score)
    
    # 切回原始目录
    os.chdir(ORIGINAL_CWD)
    
    # 合并所有掩码（逻辑或）
    if len(all_masks) > 0:
        combined_mask = np.zeros_like(all_masks[0], dtype=bool)
        for mask in all_masks:
            combined_mask = combined_mask | mask
        
        avg_score = np.mean(all_scores)
        print(f"\n✓ 分割完成！平均置信度: {avg_score:.4f}")
        return combined_mask, all_masks, all_scores
    else:
        print("\n✗ 没有生成任何掩码")
        return None, [], []


def visualize_and_save(image_path, prompts, combined_mask, individual_masks, scores, output_dir):
    """
    可视化并保存结果
    
    生成文件:
    1. prompts_vis.jpg - LVLM 输出的提示可视化
    2. segmentation_result.jpg - 分割结果叠加
    3. mask_only.png - 纯掩码图
    4. comparison.jpg - 原图与结果对比
    """
    print("\n" + "=" * 60)
    print("  Step 5: 可视化与保存")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载原图
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    width, height = image.size
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    # ----- 1. 提示可视化 -----
    prompt_vis = image.copy()
    draw = ImageDraw.Draw(prompt_vis)
    
    for i, prompt_data in enumerate(prompts):
        # 绘制 bbox（红色）
        if "bbox_2d" in prompt_data:
            bbox = prompt_data["bbox_2d"]
            draw.rectangle(bbox, outline="red", width=3)
            label = f"Target {i+1}"
            if i < len(scores):
                label += f" ({scores[i]:.2f})"
            draw.text((bbox[0], max(0, bbox[1] - 25)), label, fill="red", font=font)
        
        # 绘制 positive points（绿色）
        if "positive_points" in prompt_data:
            for j, point in enumerate(prompt_data["positive_points"]):
                px, py = int(point[0]), int(point[1])
                r = 8
                draw.ellipse([px-r, py-r, px+r, py+r], fill="lime", outline="darkgreen", width=2)
    
    prompt_vis_path = os.path.join(output_dir, "1_prompts_vis.jpg")
    prompt_vis.save(prompt_vis_path, quality=95)
    print(f"✓ 提示可视化: {prompt_vis_path}")
    
    # ----- 2. 分割结果叠加 -----
    if combined_mask is not None:
        result_overlay = image_np.copy()
        
        # 掩码区域叠加半透明蓝色
        color = np.array([30, 144, 255], dtype=np.uint8)
        alpha = 0.5
        result_overlay[combined_mask] = (
            result_overlay[combined_mask] * (1 - alpha) + color * alpha
        ).astype(np.uint8)
        
        # 绘制掩码边缘（红色）
        from scipy import ndimage
        edges = ndimage.binary_dilation(combined_mask) ^ combined_mask
        result_overlay[edges] = [255, 0, 0]
        
        result_pil = Image.fromarray(result_overlay)
        
        # 添加提示点和框
        draw_result = ImageDraw.Draw(result_pil)
        for prompt_data in prompts:
            if "bbox_2d" in prompt_data:
                draw_result.rectangle(prompt_data["bbox_2d"], outline="red", width=2)
            if "positive_points" in prompt_data:
                for point in prompt_data["positive_points"]:
                    px, py = int(point[0]), int(point[1])
                    r = 6
                    draw_result.ellipse([px-r, py-r, px+r, py+r], fill="lime", outline="darkgreen", width=2)
        
        result_path = os.path.join(output_dir, "2_segmentation_result.jpg")
        result_pil.save(result_path, quality=95)
        print(f"✓ 分割结果: {result_path}")
        
        # ----- 3. 纯掩码图 -----
        mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8))
        mask_path = os.path.join(output_dir, "3_mask_only.png")
        mask_img.save(mask_path)
        print(f"✓ 掩码图: {mask_path}")
        
        # ----- 4. 对比图 -----
        comparison = Image.new("RGB", (width * 2, height))
        comparison.paste(image, (0, 0))
        comparison.paste(result_pil, (width, 0))
        
        draw_comp = ImageDraw.Draw(comparison)
        # 添加标签背景
        draw_comp.rectangle([0, 0, 120, 30], fill="black")
        draw_comp.rectangle([width, 0, width + 150, 30], fill="black")
        draw_comp.text((10, 5), "Original", fill="white", font=font)
        draw_comp.text((width + 10, 5), "Segmentation", fill="white", font=font)
        
        comparison_path = os.path.join(output_dir, "4_comparison.jpg")
        comparison.save(comparison_path, quality=95)
        print(f"✓ 对比图: {comparison_path}")
    
    print(f"\n所有结果已保存到: {output_dir}")


def main():
    """主函数：执行完整的 LVLM + SAM2 管道"""
    
    print("\n" + "=" * 70)
    print("    Think2Seg-RS 推理管道: LVLM + SAM2.1 解耦分割")
    print("    Decoupled Reasoning-Execution Framework")
    print("=" * 70)
    
    # 检查输入
    if not os.path.exists(IMAGE_PATH):
        print(f"\n✗ 错误: 图像文件不存在: {IMAGE_PATH}")
        return
    
    # 显示配置
    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"\n配置信息:")
    print(f"  输入图像: {IMAGE_PATH}")
    print(f"  图像尺寸: {image.size}")
    print(f"  分割目标: {QUERY}")
    print(f"  输出目录: {OUTPUT_DIR}")
    
    # ===== Step 1: 加载 LVLM =====
    lvlm_model, lvlm_processor = load_lvlm()
    
    # ===== Step 2: 加载 SAM2 =====
    sam2_predictor = load_sam2()
    
    # ===== Step 3: LVLM 生成几何提示 =====
    lvlm_output = lvlm_generate_prompts(lvlm_model, lvlm_processor, IMAGE_PATH, QUERY)
    
    # 解析 JSON
    prompts = parse_json_output(lvlm_output)
    
    if not prompts:
        print("\n" + "=" * 60)
        print("  结果: 未检测到目标")
        print("=" * 60)
        print("可能的原因:")
        print("  1. 图像中没有符合描述的目标")
        print("  2. LVLM 输出格式不正确")
        print("  3. 查询描述不够清晰")
        return
    
    # 打印解析结果
    print("\n解析的几何提示:")
    for i, p in enumerate(prompts):
        print(f"  目标 {i+1}: bbox={p.get('bbox_2d')}, points={p.get('positive_points')}")
    
    # ===== Step 4: SAM2 分割 =====
    image_np = np.array(image)
    combined_mask, individual_masks, scores = sam2_segment(sam2_predictor, image_np, prompts)
    
    # ===== Step 5: 可视化与保存 =====
    if combined_mask is not None:
        visualize_and_save(IMAGE_PATH, prompts, combined_mask, individual_masks, scores, OUTPUT_DIR)
    
    # 完成
    print("\n" + "=" * 70)
    print("    ✓ 管道执行完成！")
    print("=" * 70)
    print(f"\n查看结果: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()