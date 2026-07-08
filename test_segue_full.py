# test_segue_full.py - SEGUE pipeline: reason -> segment -> verify -> refine
import sys, os, re, time, json
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'D:\afss')

import torch, numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'reasonseg_data'
OUT_DIR = Path(__file__).parent / 'reasonseg_output'
OUT_DIR.mkdir(exist_ok=True)

MAX_ITERATIONS = 1           # 最多迭代修正次数
MIN_IOU_IMPROVEMENT = 0.02   # IoU 提升低于此值就停止

# ======================== Prompts ========================

SEGUE_INITIAL_PROMPT = """You are a visual reasoning agent for precise segmentation.

Given an image and query, output geometric prompts that SAM2 will use to segment the target.

Output format:
<REASON>
[Step-by-step reasoning: what is the target? Where is it? What does it look like?
Use spatial references like "left of", "above", "at the center" etc.]
</REASON>
<SEG_BOX>
x1 y1 x2 y2
</SEG_BOX>
<SEG_POINTS>
px1 py1
px2 py2
</SEG_POINTS>

Rules:
- Box: tight bounding box. Format: top-left-x top-left-y bottom-right-x bottom-right-y
- Points: two (x,y) that are definitely INSIDE the target
- All coordinates in pixels, relative to the provided image dimensions
- Use spaces between numbers, NOT commas"""


SEGUE_VERIFY_PROMPT = """You previously segmented the target for the query:
"{query}"

Your previous geometric prompts:
- Bbox: [{bbox_x1:.0f}, {bbox_y1:.0f}, {bbox_x2:.0f}, {bbox_y2:.0f}]
- Key points: {pts_text}

The resulting mask from SAM2:
- Area: {area} pixels ({coverage:.1%} of the entire image)
- Centroid at: ({cx:.0f}, {cy:.0f})
- SAM2 confidence: {score:.3f}

{iteration_hint}

Now re-examine the image. Think about:
1. Does this mask cover the EXACT object from the query?
2. Is it too big (covering unrelated things) or too small (missing parts)?
3. Is it on the WRONG object entirely?

Output your verdict and (if wrong) improved coordinates:

<VERIFY>
YES or NO with specific reasoning about what the mask got right/wrong.
</VERIFY>
<REASON>
[If NO: step-by-step reasoning for finding the correct target]
</REASON>
<SEG_BOX>
x1 y1 x2 y2
</SEG_BOX>
<SEG_POINTS>
px1 py1
px2 py2
</SEG_POINTS>

CRITICAL: Output all 4 tags. If YES, use empty <SEG_BOX></SEG_BOX>.
Use spaces between numbers, NOT commas."""


# ======================== Model Loading ========================

print(f"[1/3] Loading Qwen2.5-VL-7B (int4)... GPU free: {torch.cuda.mem_get_info()[0]/1e9:.2f}GB")
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
import torch as _torch
_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=_torch.float16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4',
)
lvlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'D:/models/Qwen2.5-VL-7B-Instruct', torch_dtype=_torch.float16,
    device_map='auto', attn_implementation='sdpa',
    quantization_config=_bnb_config, local_files_only=True,
).eval()
proc = AutoProcessor.from_pretrained('D:/models/Qwen2.5-VL-7B-Instruct', local_files_only=True)
print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

print(f"[2/3] Loading SAM2 (CPU)...")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sam2_model = build_sam2(
    config_file='configs/sam2.1/sam2.1_hiera_b+.yaml',
    ckpt_path=r'D:\pycharm_projects\NOTRAING\NOTRAING\checkpoints\sam2.1_hiera_base_plus.pt',
    device='cpu',
)
sam2_pred = SAM2ImagePredictor(sam2_model)
print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

print(f"[3/3] Loading data...")
samples = []
for i in range(1000):  # 扫描所有可能的 idx
    ip = DATA_DIR / f'{i:04d}.jpg'; mp = DATA_DIR / f'{i:04d}_mask.npy'; jp = DATA_DIR / f'{i:04d}.json'
    if ip.exists() and mp.exists():
        samples.append({'image': Image.open(ip).convert('RGB'), 'mask': np.load(mp), 'meta': json.load(open(jp))})
print(f"  {len(samples)} samples")

from qwen_vl_utils import process_vision_info


# ======================== Helpers ========================

def qwen_chat(image, prompt_text, max_tokens=256):
    """Qwen single-turn chat with image"""
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
    """Parse bbox and points from Qwen output text"""
    text = text.replace(',', ' ')
    box_m = re.search(r'<SEG_BOX>\s*([\d\s\.]+)\s*</SEG_BOX>', text)
    pts_m = re.search(r'<SEG_POINTS>\s*([\d\s\.]+)\s*</SEG_POINTS>', text)
    verify_m = re.search(r'<VERIFY>\s*([\s\S]*?)\s*</VERIFY>', text, re.IGNORECASE)
    reason_m = re.search(r'<REASON>\s*([\s\S]*?)\s*</REASON>', text, re.IGNORECASE)

    result = {'bbox': None, 'points': [], 'verify': '', 'reason': ''}
    if verify_m: result['verify'] = verify_m.group(1).strip()
    if reason_m: result['reason'] = reason_m.group(1).strip()

    if box_m:
        nums = [float(x) for x in re.findall(r'\d+\.?\d*', box_m.group(1))]
        if len(nums) >= 4:
            result['bbox'] = nums[:4]

    if pts_m:
        pnums = [float(x) for x in re.findall(r'\d+\.?\d*', pts_m.group(1))]
        if len(pnums) >= 2:
            result['points'] = [(pnums[i], pnums[i+1]) for i in range(0, len(pnums)-1, 2)][:2]

    return result


def sam2_segment(image_np, bbox_orig, pts_orig):
    """SAM2推理"""
    sam2_pred.set_image(image_np)
    pt_arr = np.array(pts_orig, dtype=np.float32) if pts_orig else None
    pt_labels = np.ones(len(pts_orig), dtype=np.int32) if pts_orig else None
    box_arr = np.array(bbox_orig, dtype=np.float32) if bbox_orig else None
    masks, scores, _ = sam2_pred.predict(
        point_coords=pt_arr, point_labels=pt_labels, box=box_arr, multimask_output=False)
    return masks[0].astype(bool), float(scores[0])


def mask_metrics(mask):
    """计算mask的基础指标"""
    if mask is None or mask.sum() == 0:
        return {'area': 0, 'coverage': 0, 'cx': 0, 'cy': 0}
    H, W = mask.shape
    area = int(mask.sum())
    ys, xs = np.where(mask)
    return {'area': area, 'coverage': area / (H * W), 'cx': xs.mean(), 'cy': ys.mean()}


def compute_giou(pred, gt):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / max(union, 1)


SEGUE_VERIFY_CROP = (
    'You are verifying a segmentation. You see a ZOOMED-IN CROP of the bounding box region.\n\n'
    'Query: "{query}"\n'
    'Mask stats: area={area}px ({coverage:.1%} of image), SAM2 confidence={score:.3f}\n\n'
    'Is the bbox correctly enclosing the target? If not, provide improved coordinates.\n\n'
    '<VERIFY>YES or NO</VERIFY>\n'
    '<REASON>reasoning</REASON>\n'
    '<SEG_BOX>x1 y1 x2 y2</SEG_BOX>\n'
    '<SEG_POINTS>px1 py1 px2 py2</SEG_POINTS>\n\n'
    'Output all 4 tags. Spaces NOT commas. If YES use empty <SEG_BOX></SEG_BOX>.'
)


SEGUE_VERIFY_HYBRID = (
    'Verify segmentation. Image shows YELLOW OUTLINE=mask boundary, GREEN box=bbox, CROSSHAIRS=key points.\n\n'
    'Query: "{query}"\n'
    'Mask: area={area}px ({coverage:.1%}), conf={score:.3f}\n\n'
    'Does the YELLOW outline enclose the correct target?\n\n'
    '<VERIFY>YES or NO with reasoning</VERIFY>\n'
    '<REASON>refined reasoning if NO</REASON>\n'
    '<SEG_BOX>x1 y1 x2 y2</SEG_BOX>\n'
    '<SEG_POINTS>px1 py1 px2 py2</SEG_POINTS>\n\n'
    'Output all 4 tags. Spaces NOT commas. If YES use empty <SEG_BOX></SEG_BOX>.'
)


def make_outline_hybrid(original_img, mask, bbox, points, infer_size):
    """原图 + mask黄色轮廓(纯numpy) + bbox绿框 + 关键点十字"""
    w_orig, h_orig = original_img.size
    w_infer, h_infer = infer_size
    img = original_img.resize((w_infer, h_infer), Image.BILINEAR).convert('RGBA')
    if mask.shape != (h_infer, w_infer):
        mask_rs = np.array(Image.fromarray(mask).resize((w_infer, h_infer), Image.NEAREST))
    else:
        mask_rs = mask
    sc_x, sc_y = w_infer / w_orig, h_infer / h_orig

    if mask_rs.sum() > 0:
        dilated = np.zeros_like(mask_rs)
        dilated[:-1, :] |= mask_rs[1:, :]
        dilated[1:, :] |= mask_rs[:-1, :]
        dilated[:, :-1] |= mask_rs[:, 1:]
        dilated[:, 1:] |= mask_rs[:, :-1]
        dilated |= mask_rs
        edge = dilated.astype(np.uint8) - mask_rs.astype(np.uint8)
        e = np.zeros((h_infer, w_infer, 4), dtype=np.uint8)
        e[edge > 0] = [255, 220, 0, 255]
        img = Image.alpha_composite(img, Image.fromarray(e, mode='RGBA'))

    draw = ImageDraw.Draw(img)
    bx = [max(0, min(bbox[0]*sc_x, w_infer-1)), max(0, min(bbox[1]*sc_y, h_infer-1)),
          max(1, min(bbox[2]*sc_x, w_infer)), max(1, min(bbox[3]*sc_y, h_infer))]
    # Ensure x1 < x2 and y1 < y2
    bx[0], bx[2] = min(bx[0], bx[2]-1), max(bx[0]+1, bx[2])
    bx[1], bx[3] = min(bx[1], bx[3]-1), max(bx[1]+1, bx[3])
    draw.rectangle(bx, outline=(0, 255, 60), width=3)
    for px, py in points:
        x = max(0, min(px*sc_x, w_infer-1))
        y = max(0, min(py*sc_y, h_infer-1))
        L = 6
        draw.line([(max(0,x-L), y), (min(w_infer-1,x+L), y)], fill=(80, 180, 255), width=2)
        draw.line([(x, max(0,y-L)), (x, min(h_infer-1,y+L))], fill=(80, 180, 255), width=2)
    return img.convert('RGB')


def make_verification_image(original_img, mask, bbox, points, infer_size):
    """Overlay: semi-transparent red mask + yellow edge + green bbox + crosshairs"""
    from PIL import ImageDraw, ImageFont

    w_orig, h_orig = original_img.size
    w_infer, h_infer = infer_size

    img = original_img.resize((w_infer, h_infer), Image.BILINEAR).convert('RGBA')

    if mask.shape != (h_infer, w_infer):
        mask_rs = np.array(Image.fromarray(mask).resize((w_infer, h_infer), Image.NEAREST))
    else:
        mask_rs = mask

    sc_x = w_infer / w_orig
    sc_y = h_infer / h_orig

    # ---- mask 填充：红色半透明 alpha=90 (看得见但不太遮挡) ----
    fill = np.zeros((h_infer, w_infer, 4), dtype=np.uint8)
    fill[mask_rs] = [255, 40, 40, 90]
    img = Image.alpha_composite(img, Image.fromarray(fill, mode='RGBA'))

    # ---- mask 轮廓：亮黄色 2px 边 + 红色外光晕 ----
    if mask_rs.sum() > 0:
        # Pure numpy dilation (no scipy)
        def dilate(m, n):
            d = m.copy()
            for _ in range(n):
                d2 = np.zeros_like(d)
                d2[:-1,:] |= d[1:,:]; d2[1:,:] |= d[:-1,:]
                d2[:,:-1] |= d[:,1:]; d2[:,1:] |= d[:,:-1]; d |= d2
            return d
        glow = dilate(mask_rs, 2).astype(np.uint8) - mask_rs.astype(np.uint8)
        glow = glow > 0
        g = np.zeros((h_infer, w_infer, 4), dtype=np.uint8)
        g[glow] = [255, 255, 0, 180]
        img = Image.alpha_composite(img, Image.fromarray(g, mode='RGBA'))
        edge = dilate(mask_rs, 1).astype(np.uint8) - mask_rs.astype(np.uint8)
        edge = edge > 0
        e = np.zeros((h_infer, w_infer, 4), dtype=np.uint8)
        e[edge] = [255, 0, 0, 255]
        img = Image.alpha_composite(img, Image.fromarray(e, mode='RGBA'))

    # ---- bbox: 绿色粗框 ----
    draw = ImageDraw.Draw(img)
    bx = [bbox[0]*sc_x, bbox[1]*sc_y, bbox[2]*sc_x, bbox[3]*sc_y]
    draw.rectangle(bx, outline=(0, 255, 60), width=4)

    # ---- 关键点: 大十字 ----
    for i, (px, py) in enumerate(points):
        x, y = px*sc_x, py*sc_y
        L = 10
        c = (80, 180, 255) if i == 0 else (255, 150, 30)
        draw.line([(x-L,y), (x+L,y)], fill=c, width=2)
        draw.line([(x,y-L), (x,y+L)], fill=c, width=2)
        draw.ellipse([x-3,y-3, x+3,y+3], fill=(255,255,255))

    # ---- 图例 ----
    try: font = ImageFont.truetype("arial.ttf", 12)
    except: font = ImageFont.load_default()

    bar_h = 48
    draw.rectangle([0, 0, w_infer, bar_h], fill=(0, 0, 0, 210))
    draw.text((6, 3),  "RED area = current mask", fill=(255, 60, 60), font=font)
    draw.text((200, 3), "YELLOW edge = mask boundary", fill=(255, 255, 0), font=font)
    draw.text((6, 18), "GREEN box = bbox", fill=(0, 255, 60), font=font)
    draw.text((200, 18), "CROSSHAIRS = key points inside target", fill=(80, 180, 255), font=font)
    draw.text((6, 33), "RED edge = boundary detail", fill=(255, 0, 0), font=font)

    return img.convert('RGB')


# ======================== Main Loop ========================

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=100, help='Number of test samples')
parser.add_argument('--start', type=int, default=0, help='Starting sample index')
parser.add_argument('--hybrid', action='store_true', help='Use hybrid verification (outline+text)')
parser.add_argument('--crop', action='store_true', help='Use crop-zoom verification')
args = parser.parse_args()

N = min(args.num_samples, len(samples) - args.start)
sample_indices = list(range(args.start, args.start + N))

# 自动续跑：检查上次已完成的样本，跳过
_results_file = OUT_DIR / 'segue_results.json'
_done_indices = set()
if _results_file.exists() and args.start > 0:
    try:
        with open(_results_file) as _f:
            _prev = _json.load(_f)
        for _r in _prev:
            if 'idx' in _r:
                _done_indices.add(_r['idx'])
            elif 'sample_idx' in _r:
                _done_indices.add(_r['sample_idx'])
    except: pass

    _before = len(sample_indices)
    sample_indices = [i for i in sample_indices if i not in _done_indices]
    print(f"  Skipping {_before - len(sample_indices)} already processed samples (resume)")

print(f"\n{'='*65}")
print(f"  SEGUE Full Pipeline: Reasoning → Segment → Verify → Refine")
print(f"  {len(sample_indices)} samples to process (indices {sample_indices[0] if sample_indices else 'N/A'}-{sample_indices[-1] if sample_indices else 'N/A'}), "
      f"max {MAX_ITERATIONS} refinements")
print(f"{'='*65}")

all_results = []

for idx in sample_indices:
    sample = samples[idx]
    img = sample['image']
    gt_mask = sample['mask']
    query = sample['meta']['text']
    w_orig, h_orig = img.size

    # Resize for Qwen
    max_size = 800
    scale = min(max_size / max(w_orig, h_orig), 1.0)
    w_infer = max(64, int(w_orig * scale))
    h_infer = max(64, int(h_orig * scale))
    img_small = img.resize((w_infer, h_infer), Image.BILINEAR)
    img_np = np.array(img)

    print(f"\n{'─'*65}")
    print(f"[Sample {idx}] {query[:90]}")
    print(f"  Image: {w_orig}x{h_orig} → Qwen input: {w_infer}x{h_infer}")

    # ================================================================
    # ITERATION 0: Initial reasoning + segmentation
    # ================================================================
    iter_log = []
    t_start = time.time()

    prompt0 = SEGUE_INITIAL_PROMPT.format(w=w_infer, h=h_infer, query=query)
    out0 = qwen_chat(img_small, prompt0, max_tokens=300)
    parsed0 = parse_geometric(out0)

    if parsed0['bbox'] is None:
        print(f"  [FAIL] Iter 0: No bbox parsed")
        all_results.append({'idx': idx, 'query': query, 'gIoU_initial': 0, 'gIoU_best': 0,
                            'improvement': 0, 'total_iterations': 0, 'time': 0,
                            'best_iter': 0, 'converged': False})
        continue

    # Scale to original
    s = 1.0 / scale
    bbox_cur = [parsed0['bbox'][0]*s, parsed0['bbox'][1]*s, parsed0['bbox'][2]*s, parsed0['bbox'][3]*s]
    pts_cur = [(p[0]*s, p[1]*s) for p in parsed0['points']]

    mask_cur, score_cur = sam2_segment(img_np, bbox_cur, pts_cur)
    metrics_cur = mask_metrics(mask_cur)
    giou_cur = compute_giou(mask_cur, gt_mask)

    iter_log.append({'iter': 0, 'bbox': bbox_cur.copy(), 'points': pts_cur.copy(),
                     'score': score_cur, 'area': metrics_cur['area'], 'gIoU': giou_cur,
                     'reason': parsed0['reason'][:150]})

    print(f"  [Iter 0] bbox={[f'{b:.0f}' for b in bbox_cur]}, "
          f"score={score_cur:.3f}, area={metrics_cur['area']}px, gIoU={giou_cur:.4f}")

    # ================================================================
    # ITERATIONS 1..N: Verify → Refine → Re-segment
    # ================================================================
    prev_giou = giou_cur
    best_mask, best_giou = mask_cur.copy(), giou_cur
    best_iter = 0
    converged = False

    for it in range(1, MAX_ITERATIONS + 1):
        if converged:
            break

        # Build iteration hint
        if it == 1:
            hint = "This is your FIRST refinement attempt. Check carefully whether the mask correctly covers the target described in the query."
        elif it == MAX_ITERATIONS:
            hint = "This is your LAST refinement attempt. Make your best final adjustment."
        else:
            hint = f"Refinement attempt {it}. Try to improve upon the previous mask."

        # 验证：混合模式(轮廓叠加+文本) 或 纯文本模式
        if args.crop:
            # Crop-zoom: 裁剪 bbox 区域，放大到 512x512
            x1 = int(max(0, bbox_cur[0] - 0.15*(bbox_cur[2]-bbox_cur[0])))
            y1 = int(max(0, bbox_cur[1] - 0.15*(bbox_cur[3]-bbox_cur[1])))
            x2 = int(min(w_orig, bbox_cur[2] + 0.15*(bbox_cur[2]-bbox_cur[0])))
            y2 = int(min(h_orig, bbox_cur[3] + 0.15*(bbox_cur[3]-bbox_cur[1])))
            if x2 <= x1+10: x1, x2 = 0, w_orig
            if y2 <= y1+10: y1, y2 = 0, h_orig
            verify_img = img.crop((x1, y1, x2, y2)).resize((512, 512), Image.BILINEAR)
            verify_prompt = SEGUE_VERIFY_CROP.format(
                query=query, area=metrics_cur['area'], coverage=metrics_cur['coverage'],
                cx=metrics_cur['cx'], cy=metrics_cur['cy'], score=score_cur,
            )
            out_refine = qwen_chat(verify_img, verify_prompt, max_tokens=400)

        elif args.hybrid:
            # 生成 mask 轮廓叠加图
            verify_img = make_outline_hybrid(img, mask_cur, bbox_cur, pts_cur, (w_infer, h_infer))
            verify_prompt = SEGUE_VERIFY_HYBRID.format(
                query=query, area=metrics_cur['area'], coverage=metrics_cur['coverage'],
                cx=metrics_cur['cx'], cy=metrics_cur['cy'], score=score_cur,
            )
            out_refine = qwen_chat(verify_img, verify_prompt, max_tokens=400)
        else:
            # 计算框的详细指标
            bbox_w = bbox_cur[2] - bbox_cur[0]
            bbox_h = bbox_cur[3] - bbox_cur[1]
            bbox_w_pct = bbox_w / w_orig * 100
            bbox_h_pct = bbox_h / h_orig * 100
            cx_pct = (bbox_cur[0] + bbox_cur[2]) / 2 / w_orig * 100
            cy_pct = (bbox_cur[1] + bbox_cur[3]) / 2 / h_orig * 100
            bbox_area = max(bbox_w * bbox_h, 1)
            fill_ratio = metrics_cur['area'] / bbox_area

            pts_text = "none" if not pts_cur else " ".join([f"({p[0]:.0f},{p[1]:.0f})" for p in pts_cur])

            verify_prompt = SEGUE_VERIFY_PROMPT.format(
                query=query,
                bbox_x1=bbox_cur[0], bbox_y1=bbox_cur[1],
                bbox_x2=bbox_cur[2], bbox_y2=bbox_cur[3],
                pts_text=pts_text,
                area=metrics_cur['area'],
                coverage=metrics_cur['coverage'],
                cx=metrics_cur['cx'],
                cy=metrics_cur['cy'],
                score=score_cur,
                iteration_hint=hint,
            )
            out_refine = qwen_chat(img_small, verify_prompt, max_tokens=400)
        parsed_refine = parse_geometric(out_refine)

        # Check if Qwen thinks it's correct
        verify_text = parsed_refine['verify'].upper()
        if verify_text.startswith('YES') and parsed_refine['bbox'] is None:
            print(f"  [Iter {it}] [OK] Qwen verified: mask is CORRECT. Stopping.")
            converged = True
            break

        if parsed_refine['bbox'] is None:
            print(f"  [Iter {it}] [FAIL] No refined bbox. Stopping refinement.")
            break

        # Scale refined coordinates
        bbox_new = [parsed_refine['bbox'][0]*s, parsed_refine['bbox'][1]*s,
                    parsed_refine['bbox'][2]*s, parsed_refine['bbox'][3]*s]
        pts_new = [(p[0]*s, p[1]*s) for p in parsed_refine['points']]

        # Re-segment
        mask_new, score_new = sam2_segment(img_np, bbox_new, pts_new)
        metrics_new = mask_metrics(mask_new)
        giou_new = compute_giou(mask_new, gt_mask)

        improved = giou_new > prev_giou + MIN_IOU_IMPROVEMENT
        sign = "▲" if improved else "▼"

        print(f"  [Iter {it}] {sign} bbox={[f'{b:.0f}' for b in bbox_new]}, "
              f"score={score_new:.3f}, area={metrics_new['area']}px, gIoU={giou_new:.4f} "
              f"({'improved' if improved else 'no improvement'})")

        iter_log.append({'iter': it, 'bbox': bbox_new.copy(), 'points': pts_new.copy(),
                         'score': score_new, 'area': metrics_new['area'], 'gIoU': giou_new,
                         'reason': parsed_refine['reason'][:150],
                         'verify': parsed_refine['verify'][:150],
                         'improved': improved})

        # Update tracking
        bbox_cur, pts_cur = bbox_new, pts_new
        mask_cur, score_cur = mask_new, score_new
        metrics_cur = metrics_new
        prev_giou = giou_new

        if giou_new > best_giou:
            best_giou = giou_new
            best_mask = mask_new.copy()
            best_iter = it

        # Stop if no improvement for 2 consecutive
        if it >= 2 and not improved and not iter_log[-2].get('improved', False):
            print(f"  → No improvement for 2 iterations. Stopping.")
            converged = True

    t_total = time.time() - t_start

    # ================================================================
    # Summary per sample
    # ================================================================
    print(f"  ─{'─'*40}")
    print(f"  Final: best gIoU={best_giou:.4f} (iter {best_iter}), "
          f"initial={iter_log[0]['gIoU']:.4f}, "
          f"delta={best_giou - iter_log[0]['gIoU']:+.4f}, "
          f"time={t_total:.1f}s")

    all_results.append({
        'idx': idx,
        'query': query,
        'gIoU_initial': iter_log[0]['gIoU'],
        'gIoU_best': best_giou,
        'best_iter': best_iter,
        'total_iterations': len(iter_log),
        'improvement': best_giou - iter_log[0]['gIoU'],
        'converged': converged,
        'time': t_total,
        'iter_log': iter_log,
    })

    # Visualization: initial vs best mask
    if best_mask.shape != (h_orig, w_orig):
        pred_vis = np.array(Image.fromarray(best_mask).resize((w_orig, h_orig), Image.NEAREST))
        init_vis = np.array(Image.fromarray(
            Image.fromarray(iter_log[0].get('mask_init', best_mask) if False else best_mask)
        ).resize((w_orig, h_orig), Image.NEAREST))
    else:
        pred_vis = best_mask

    # Initial mask visualization
    mask_init = iter_log[0]
    init_mask_data = mask_cur if len(iter_log) == 1 else None
    # Re-generate initial mask for comparison
    init_bbox = [iter_log[0]['bbox'][0], iter_log[0]['bbox'][1],
                 iter_log[0]['bbox'][2], iter_log[0]['bbox'][3]]
    init_pts = iter_log[0]['points']
    mask_init_arr, _ = sam2_segment(img_np, init_bbox, init_pts)
    if mask_init_arr.shape != (h_orig, w_orig):
        mask_init_arr = np.array(Image.fromarray(mask_init_arr).resize((w_orig, h_orig), Image.NEAREST))
    if best_mask.shape != (h_orig, w_orig):
        pred_vis = np.array(Image.fromarray(best_mask).resize((w_orig, h_orig), Image.NEAREST))

    # Composite: GT=green, Initial=blue, Best=red
    vis = np.array(img).copy()[:,:,:3]
    overlay = vis.copy()
    overlay[gt_mask] = [0, 255, 0]      # GT: green
    overlay[mask_init_arr] = [0, 100, 255]  # Initial: blue
    overlay[pred_vis] = [255, 0, 0]     # Best: red
    blended = (vis * 0.4 + overlay * 0.6).astype(np.uint8)

    from PIL import ImageDraw, ImageFont
    result_img = Image.fromarray(blended)
    draw = ImageDraw.Draw(result_img)
    try: font = ImageFont.truetype("arial.ttf", 14)
    except: font = ImageFont.load_default()
    draw.rectangle([0, 0, result_img.width, 45], fill=(0,0,0,200))
    draw.text((5, 2), f"GT=green Init=blue Best=red | gIoU: {iter_log[0]['gIoU']:.3f} → {best_giou:.3f}", fill='white', font=font)
    draw.text((5, 24), f"Query: {query[:100]}", fill='lightgray', font=font)
    result_img.save(OUT_DIR / f'segue_full_{idx:02d}.jpg')

# ======================== Final Summary ========================
print(f"\n{'='*65}")
print(f"  SEGUE Full Pipeline Results")
print(f"{'='*65}")
print(f"  {'Sample':<8s} {'Init gIoU':>10s} {'Best gIoU':>10s} {'Δ':>8s} {'Iters':>6s} {'Time':>7s}")
print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*6} {'─'*7}")

init_gious, best_gious, improvements = [], [], []
for i, r in enumerate(all_results):
    di = r.get('gIoU_initial', 0)
    db = r.get('gIoU_best', 0)
    imp = r.get('improvement', 0)
    print(f"  [{i}]     {di:10.4f} {db:10.4f} {imp:+8.4f} {r['total_iterations']:6d} {r['time']:6.1f}s")
    init_gious.append(di)
    best_gious.append(db)
    improvements.append(imp)

print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*6} {'─'*7}")
print(f"  Mean:   {np.mean(init_gious):10.4f} {np.mean(best_gious):10.4f} {np.mean(improvements):+8.4f}")
print(f"\n  Improved samples: {sum(1 for x in improvements if x > 0.01)}/{len(all_results)}")
print(f"  Avg iterations:   {np.mean([r['total_iterations'] for r in all_results]):.1f}")
print(f"  Visualizations:   {OUT_DIR}/")

# Save results JSON
import json as _json
def to_native(v):
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, (np.ndarray,)): return v.tolist()
    return v

save_data = []
for r in all_results:
    d = {k: to_native(v) for k, v in r.items() if k != 'iter_log'}
    d['iter_log'] = [{kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else bool(vv) if isinstance(vv, np.bool_) else vv) for kk, vv in it.items()} for it in r.get('iter_log', [])]
    save_data.append(d)
with open(OUT_DIR / 'segue_results.json', 'w') as f:
    _json.dump(save_data, f, indent=2, ensure_ascii=False)
# Also save a partial with just this batch (for resume inspection)
if args.start > 0:
    with open(OUT_DIR / f'segue_results_batch_{args.start}.json', 'w') as f:
        _json.dump(save_data, f, indent=2, ensure_ascii=False)
print(f"  Results saved: {OUT_DIR / 'segue_results.json'}")
