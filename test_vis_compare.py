"""
test_vis_compare.py — 对比三种可视化验证方案
============================================
A: 纯文本（当前最佳基线）  — 只给数字指标
B: 轮廓线叠加             — 原图+mask黄色轮廓+bbox绿框
C: 抠图展示               — 原图 + 抠出的目标区域(黑底)
"""
import sys, os, re, time, json
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'D:\afss')

import torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from scipy import ndimage

DATA_DIR = Path(__file__).parent / 'reasonseg_data'
OUT_DIR = Path(__file__).parent / 'reasonseg_output'
OUT_DIR.mkdir(exist_ok=True)

# ===================== 两种可视化方案 =====================

def make_outline_image(original_img, mask, bbox, points, infer_size):
    """
    方案B: 只画轮廓，不填充。
    - mask边界用亮黄色粗轮廓
    - bbox用绿色框
    - 关键点用十字
    """
    w_orig, h_orig = original_img.size
    w_infer, h_infer = infer_size

    img = original_img.resize((w_infer, h_infer), Image.BILINEAR).convert('RGBA')
    if mask.shape != (h_infer, w_infer):
        mask_rs = np.array(Image.fromarray(mask).resize((w_infer, h_infer), Image.NEAREST))
    else:
        mask_rs = mask

    sc_x = w_infer / w_orig
    sc_y = h_infer / h_orig

    # Mask 轮廓：亮黄色 3px
    if mask_rs.sum() > 0:
        edge = ndimage.binary_dilation(mask_rs, iterations=1).astype(np.uint8) - mask_rs.astype(np.uint8)
        edge = edge > 0
        e = np.zeros((h_infer, w_infer, 4), dtype=np.uint8)
        e[edge] = [255, 220, 0, 255]  # 亮黄，不透明
        img = Image.alpha_composite(img, Image.fromarray(e, mode='RGBA'))

    draw = ImageDraw.Draw(img)

    # Bbox: 绿色
    bx = [bbox[0]*sc_x, bbox[1]*sc_y, bbox[2]*sc_x, bbox[3]*sc_y]
    draw.rectangle(bx, outline=(0, 255, 60), width=3)

    # 关键点: 十字
    for i, (px, py) in enumerate(points):
        x, y = px*sc_x, py*sc_y
        L = 8
        c = (80, 180, 255) if i == 0 else (255, 150, 30)
        draw.line([(x-L,y), (x+L,y)], fill=c, width=2)
        draw.line([(x,y-L), (x,y+L)], fill=c, width=2)

    # 图例
    try: font = ImageFont.truetype("arial.ttf", 12)
    except: font = ImageFont.load_default()
    draw.rectangle([0, 0, w_infer, 28], fill=(0, 0, 0, 200))
    draw.text((6, 6), "Yellow outline = mask boundary  |  Green box = bbox  |  Crosshairs = key points",
              fill=(255, 255, 200), font=font)

    return img.convert('RGB')


def make_cutout_image(original_img, mask, bbox, points, infer_size):
    """
    方案C: 抠图展示。
    左侧 = 原图 + bbox绿框
    右侧 = 只显示 mask 区域的内容（黑底），让 Qwen 直观看到"割出了什么"
    """
    w_orig, h_orig = original_img.size
    w_infer, h_infer = infer_size

    # Left panel: original with bbox
    left = original_img.resize((w_infer, h_infer), Image.BILINEAR).convert('RGB')
    if mask.shape != (h_infer, w_infer):
        mask_rs = np.array(Image.fromarray(mask).resize((w_infer, h_infer), Image.NEAREST))
    else:
        mask_rs = mask

    sc_x = w_infer / w_orig
    sc_y = h_infer / h_orig

    draw_left = ImageDraw.Draw(left)
    bx = [bbox[0]*sc_x, bbox[1]*sc_y, bbox[2]*sc_x, bbox[3]*sc_y]
    draw_left.rectangle(bx, outline=(0, 255, 60), width=3)
    for i, (px, py) in enumerate(points):
        x, y = px*sc_x, py*sc_y
        L = 6
        c = (80, 180, 255) if i == 0 else (255, 150, 30)
        draw_left.line([(x-L,y), (x+L,y)], fill=c, width=2)
        draw_left.line([(x,y-L), (x,y+L)], fill=c, width=2)

    # Right panel: cutout (only masked region on black background)
    img_np = np.array(original_img.resize((w_infer, h_infer), Image.BILINEAR))
    cutout = np.zeros_like(img_np)
    cutout[mask_rs] = img_np[mask_rs]
    right = Image.fromarray(cutout)

    # Composite: left | right
    gap = 4
    total_w = w_infer * 2 + gap
    total_h = h_infer + 30
    combined = Image.new('RGB', (total_w, total_h), (30, 30, 30))
    combined.paste(left, (0, 30))
    combined.paste(right, (w_infer + gap, 30))

    draw = ImageDraw.Draw(combined)
    try: font = ImageFont.truetype("arial.ttf", 13)
    except: font = ImageFont.load_default()
    draw.text((6, 6), "LEFT: original + bbox  |  RIGHT: extracted region (what SAM2 segmented)",
              fill=(200, 200, 200), font=font)
    # Labels
    draw.text((w_infer//2 - 30, 8), "ORIGINAL", fill=(0, 255, 60), font=font)
    draw.text((w_infer + gap + w_infer//2 - 30, 8), "EXTRACTED", fill=(255, 220, 0), font=font)

    return combined


# ===================== 测试对比 =====================

print("[1/2] Loading Qwen2.5-VL-3B...")
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
lvlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype=torch.float16,
    device_map='auto', attn_implementation='sdpa', local_files_only=True,
).eval()
proc = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', local_files_only=True)

print("[2/2] Loading SAM2 (CPU)...")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sam2_model = build_sam2(
    config_file='configs/sam2.1/sam2.1_hiera_b+.yaml',
    ckpt_path=r'D:\pycharm_projects\NOTRAING\NOTRAING\checkpoints\sam2.1_hiera_base_plus.pt',
    device='cpu',
)
sam2_pred = SAM2ImagePredictor(sam2_model)

# Load 3 samples that previously had mixed results
samples = []
for i in [0, 1, 2, 4]:  # skip sample 3 (failed)
    ip = DATA_DIR / f'{i:04d}.jpg'; mp = DATA_DIR / f'{i:04d}_mask.npy'; jp = DATA_DIR / f'{i:04d}.json'
    if ip.exists():
        samples.append({'image': Image.open(ip).convert('RGB'), 'mask': np.load(mp),
                        'meta': json.load(open(jp)), 'idx': i})

from qwen_vl_utils import process_vision_info

# ---- Shared prompt ----
VERIFY_B = """You are verifying a segmentation result.
Query: "{query}"

This image shows a YELLOW OUTLINE around the segmented region.
The GREEN box is the bounding box. CROSSHAIRS are key points.

Your previous attempt had these mask stats:
- Area: {area} pixels ({coverage:.1%} of image), Centroid: ({cx:.0f},{cy:.0f}), SAM2 conf: {score:.3f}

Does the YELLOW outline correctly enclose the target from the query?
<VERIFY>YES or NO with reasoning</VERIFY>
<REASON>refined reasoning if NO</REASON>
<SEG_BOX>x1 y1 x2 y2</SEG_BOX>
<SEG_POINTS>px1 py1 px2 py2</SEG_POINTS>
CRITICAL: Output all 4 tags. Use spaces between numbers."""

VERIFY_C = '''You are verifying a segmentation result.
Query: "{query}"

The LEFT image shows the original scene with a GREEN bbox.
The RIGHT image shows ONLY what SAM2 extracted (on black background).

Look at the RIGHT panel: does it show the EXACT object from the query?
- If the right panel shows the wrong thing, something else, or too much → NO
- If it shows exactly the right target → YES

Mask stats: area={area}px ({coverage:.1%}), centroid=({cx:.0f},{cy:.0f}), conf={score:.3f}

<VERIFY>YES or NO with reasoning</VERIFY>
<REASON>refined reasoning if NO</REASON>
<SEG_BOX>x1 y1 x2 y2</SEG_BOX>
<SEG_POINTS>px1 py1 px2 py2</SEG_POINTS>
CRITICAL: Output all 4 tags. Spaces between numbers, NO commas.'''


def qwen_chat(image, prompt_text, max_tokens=300):
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': prompt_text},
    ]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                  padding=True, return_tensors='pt').to(lvlm.device)
    with torch.no_grad():
        out_ids = lvlm.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    out_text = proc.batch_decode(out_ids, skip_special_tokens=True)[0]
    if 'assistant' in out_text:
        out_text = out_text.split('assistant')[-1].strip()
    return out_text


def parse_geometric(text):
    text = text.replace(',', ' ')
    box_m = re.search(r'<SEG_BOX>\s*([\d\s\.]+)\s*</SEG_BOX>', text)
    pts_m = re.search(r'<SEG_POINTS>\s*([\d\s\.]+)\s*</SEG_POINTS>', text)
    result = {'bbox': None, 'points': []}
    if box_m:
        nums = [float(x) for x in re.findall(r'\d+\.?\d*', box_m.group(1))]
        if len(nums) >= 4: result['bbox'] = nums[:4]
    if pts_m:
        pnums = [float(x) for x in re.findall(r'\d+\.?\d*', pts_m.group(1))]
        result['points'] = [(pnums[i], pnums[i+1]) for i in range(0, len(pnums)-1, 2)][:2]
    return result


def sam2_segment(image_np, bbox, pts):
    sam2_pred.set_image(image_np)
    pt_arr = np.array(pts, dtype=np.float32) if pts else None
    pt_labels = np.ones(len(pts), dtype=np.int32) if pts else None
    box_arr = np.array(bbox, dtype=np.float32) if bbox else None
    masks, scores, _ = sam2_pred.predict(
        point_coords=pt_arr, point_labels=pt_labels, box=box_arr, multimask_output=False)
    return masks[0].astype(bool), float(scores[0])


def compute_giou(pred, gt):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / max(union, 1)


# ===================== 对比测试 =====================

print(f"\n{'='*55}")
print(f"  Comparing 2 visualization schemes on {len(samples)} samples")
print(f"{'='*55}")

for sample in samples:
    img = sample['image']
    gt_mask = sample['mask']
    query = sample['meta']['text']
    w_orig, h_orig = img.size
    max_sz = 800
    scale = min(max_sz / max(w_orig, h_orig), 1.0)
    w_inf = max(64, int(w_orig * scale))
    h_inf = max(64, int(h_orig * scale))
    img_small = img.resize((w_inf, h_inf), Image.BILINEAR)
    img_np = np.array(img)

    print(f"\n--- Sample {sample['idx']}: {query[:70]}... ---")

    # Get initial bbox from Qwen (same prompt as before)
    init_prompt = (
        "You are a visual reasoning agent. Given the image and query, output:\n"
        "<REASON>step-by-step reasoning</REASON>\n"
        "<SEG_BOX>x1 y1 x2 y2</SEG_BOX>\n"
        "<SEG_POINTS>px1 py1 px2 py2</SEG_POINTS>\n\n"
        f"Image: {w_inf}x{h_inf} pixels\n"
        f"Query: {query}\n"
        "Generate:"
    )

    out0 = qwen_chat(img_small, init_prompt, max_tokens=300)
    parsed0 = parse_geometric(out0)
    if parsed0['bbox'] is None:
        print(f"  [FAIL] No initial bbox")
        continue

    s = 1.0 / scale
    bbox0 = [parsed0['bbox'][0]*s, parsed0['bbox'][1]*s,
             parsed0['bbox'][2]*s, parsed0['bbox'][3]*s]
    pts0 = [(p[0]*s, p[1]*s) for p in parsed0['points']]

    mask0, score0 = sam2_segment(img_np, bbox0, pts0)
    giou0 = compute_giou(mask0, gt_mask)
    ys, xs = np.where(mask0) if mask0.sum() > 0 else ([0], [0])
    metrics = {'area': int(mask0.sum()), 'coverage': mask0.sum()/(mask0.shape[0]*mask0.shape[1]),
               'cx': xs.mean(), 'cy': ys.mean()}

    # ---- Test Scheme B: Outline only ----
    vis_b = make_outline_image(img, mask0, bbox0, pts0, (w_inf, h_inf))
    prompt_b = VERIFY_B.format(query=query, area=metrics['area'],
                                coverage=metrics['coverage'], cx=metrics['cx'],
                                cy=metrics['cy'], score=score0)
    out_b = qwen_chat(vis_b, prompt_b, max_tokens=300)
    parsed_b = parse_geometric(out_b)
    giou_b = giou0
    if parsed_b['bbox'] is not None:
        bb = [parsed_b['bbox'][0]*s, parsed_b['bbox'][1]*s, parsed_b['bbox'][2]*s, parsed_b['bbox'][3]*s]
        pb = [(p[0]*s, p[1]*s) for p in parsed_b['points']]
        mb, _ = sam2_segment(img_np, bb, pb)
        giou_b = compute_giou(mb, gt_mask)

    # ---- Test Scheme C: Cutout ----
    vis_c = make_cutout_image(img, mask0, bbox0, pts0, (w_inf, h_inf))
    prompt_c = VERIFY_C.format(query=query, area=metrics['area'],
                                coverage=metrics['coverage'], cx=metrics['cx'],
                                cy=metrics['cy'], score=score0)
    out_c = qwen_chat(vis_c, prompt_c, max_tokens=300)
    parsed_c = parse_geometric(out_c)
    giou_c = giou0
    if parsed_c['bbox'] is not None:
        bc = [parsed_c['bbox'][0]*s, parsed_c['bbox'][1]*s, parsed_c['bbox'][2]*s, parsed_c['bbox'][3]*s]
        pc = [(p[0]*s, p[1]*s) for p in parsed_c['points']]
        mc, _ = sam2_segment(img_np, bc, pc)
        giou_c = compute_giou(mc, gt_mask)

    # Report
    print(f"  Initial gIoU:      {giou0:.4f}")
    print(f"  Scheme B (outline): {giou_b:.4f}  delta={giou_b-giou0:+.4f}  {'OK' if parsed_b['bbox'] else 'FAIL(no bbox)'}")
    print(f"  Scheme C (cutout):  {giou_c:.4f}  delta={giou_c-giou0:+.4f}  {'OK' if parsed_c['bbox'] else 'FAIL(no bbox)'}")

    # Save vis images for inspection
    sidx = sample['idx']
    vis_b.save(OUT_DIR / f'vis_b_outline_{sidx}.jpg')
    vis_c.save(OUT_DIR / f'vis_c_cutout_{sidx}.jpg')

print(f"\nDone. Visualizations in {OUT_DIR}/")
