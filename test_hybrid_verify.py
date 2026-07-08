"""
test_hybrid_verify.py — 混合验证: mask轮廓叠加图 + 文本框数字指标
"""
import sys, os, re, time, json
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'D:\afss')

import torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'reasonseg_data'

print("[1/2] Loading Qwen...")
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
lvlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype=torch.float16,
    device_map='auto', attn_implementation='sdpa', local_files_only=True,
).eval()
proc = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', local_files_only=True)

print("[2/2] Loading SAM2...")
# Must cd to SAM2 directory for Hydra config resolution
_orig_cwd = os.getcwd()
os.chdir(r'D:\afss\sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sam2_model = build_sam2(
    config_file='configs/sam2.1/sam2.1_hiera_b+.yaml',
    ckpt_path=r'D:\pycharm_projects\NOTRAING\NOTRAING\checkpoints\sam2.1_hiera_base_plus.pt',
    device='cpu',
)
sam2_pred = SAM2ImagePredictor(sam2_model)
os.chdir(_orig_cwd)

from qwen_vl_utils import process_vision_info

# ====== Helpers ======

def make_outline_overlay(original_img, mask, bbox, points, infer_size):
    """原图 + mask黄色轮廓 + bbox绿框 + 关键点十字"""
    w_orig, h_orig = original_img.size
    w_infer, h_infer = infer_size
    img = original_img.resize((w_infer, h_infer), Image.BILINEAR).convert('RGBA')
    if mask.shape != (h_infer, w_infer):
        mask_rs = np.array(Image.fromarray(mask).resize((w_infer, h_infer), Image.NEAREST))
    else:
        mask_rs = mask
    sc_x, sc_y = w_infer / w_orig, h_infer / h_orig

    if mask_rs.sum() > 0:
        # Pure numpy edge detection (replace scipy.ndimage.binary_dilation)
        dilated = np.zeros_like(mask_rs)
        dilated[:-1, :] |= mask_rs[1:, :]    # shift up
        dilated[1:, :] |= mask_rs[:-1, :]    # shift down
        dilated[:, :-1] |= mask_rs[:, 1:]    # shift left
        dilated[:, 1:] |= mask_rs[:, :-1]    # shift right
        dilated |= mask_rs                    # include self
        edge = dilated.astype(np.uint8) - mask_rs.astype(np.uint8)
        e = np.zeros((h_infer, w_infer, 4), dtype=np.uint8)
        e[edge > 0] = [255, 220, 0, 255]
        img = Image.alpha_composite(img, Image.fromarray(e, mode='RGBA'))

    draw = ImageDraw.Draw(img)
    bx = [bbox[0]*sc_x, bbox[1]*sc_y, bbox[2]*sc_x, bbox[3]*sc_y]
    draw.rectangle(bx, outline=(0, 255, 60), width=3)

    for i, (px, py) in enumerate(points):
        x, y = px*sc_x, py*sc_y
        L = 8
        c = (80, 180, 255) if i == 0 else (255, 150, 30)
        draw.line([(x-L, y), (x+L, y)], fill=c, width=2)
        draw.line([(x, y-L), (x, y+L)], fill=c, width=2)

    try: font = ImageFont.truetype('arial.ttf', 11)
    except: font = ImageFont.load_default()
    draw.rectangle([0, 0, w_infer, 22], fill=(0, 0, 0, 200))
    draw.text((5, 3), 'Yellow=mask boundary | Green=bbox | Cross=keypoints',
              fill=(255, 255, 200), font=font)
    return img.convert('RGB')


HYBRID_PROMPT = """You are verifying a segmentation. The image shows:
- YELLOW outline = your previous mask boundary
- GREEN box = bounding box
- CROSSHAIRS = key points inside target

Mask metrics:
- Area: {area} pixels ({coverage:.1%} of image)
- Centroid: ({cx:.0f}, {cy:.0f})
- SAM2 confidence: {score:.3f}

Look at the YELLOW outline. Does it correctly enclose the target for:
"{query}"

<VERIFY>YES or NO with specific reasoning</VERIFY>
<REASON>refined reasoning if NO</REASON>
<SEG_BOX>x1 y1 x2 y2</SEG_BOX>
<SEG_POINTS>px1 py1 px2 py2</SEG_POINTS>

CRITICAL: Output ALL 4 tags. Spaces between numbers, NO commas. If YES, use empty <SEG_BOX></SEG_BOX>."""


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
    bm = re.search(r'<SEG_BOX>\s*([\d\s\.]+)\s*</SEG_BOX>', text)
    pm = re.search(r'<SEG_POINTS>\s*([\d\s\.]+)\s*</SEG_POINTS>', text)
    r = {'bbox': None, 'points': []}
    if bm:
        nums = [float(x) for x in re.findall(r'\d+\.?\d*', bm.group(1))]
        if len(nums) >= 4: r['bbox'] = nums[:4]
    if pm:
        pnums = [float(x) for x in re.findall(r'\d+\.?\d*', pm.group(1))]
        r['points'] = [(pnums[i], pnums[i+1]) for i in range(0, len(pnums)-1, 2)][:2]
    return r


def sam2_seg(img_np, bbox, pts):
    sam2_pred.set_image(img_np)
    pa = np.array(pts, dtype=np.float32) if pts else None
    pl = np.ones(len(pts), dtype=np.int32) if pts else None
    ba = np.array(bbox, dtype=np.float32) if bbox else None
    m, s, _ = sam2_pred.predict(point_coords=pa, point_labels=pl, box=ba, multimask_output=False)
    return m[0].astype(bool), float(s[0])


def giou(pred, gt):
    i = (pred & gt).sum()
    u = (pred | gt).sum()
    return i / max(u, 1)


# ====== Load test samples ======
samples = []
for i in range(1000):
    ip = DATA_DIR / f'{i:04d}.jpg'
    mp = DATA_DIR / f'{i:04d}_mask.npy'
    jp = DATA_DIR / f'{i:04d}.json'
    if ip.exists() and mp.exists():
        samples.append({'idx': i, 'img': Image.open(ip).convert('RGB'),
                        'mask': np.load(mp), 'meta': json.load(open(jp))})

print(f"\nLoaded {len(samples)} samples. Running hybrid verification on ALL 779...\n")

results = []
for s in samples:
    img, gt, q = s['img'], s['mask'], s['meta']['text']
    w0, h0 = img.size
    sc = min(800 / max(w0, h0), 1.0)
    wi, hi = max(64, int(w0*sc)//16*16), max(64, int(h0*sc)//16*16)
    ism = img.resize((wi, hi), Image.BILINEAR)
    inp = np.array(img)

    # Initial inference
    iprompt = (
        "You are a visual reasoning agent.\n"
        f"Image: {wi}x{hi}\nQuery: {q}\n"
        "Output:\n<REASON>...</REASON>\n<SEG_BOX>x1 y1 x2 y2</SEG_BOX>\n"
        "<SEG_POINTS>px1 py1 px2 py2</SEG_POINTS>"
    )
    out0 = qwen_chat(ism, iprompt, 256)
    p0 = parse_geometric(out0)
    if not p0['bbox']:
        results.append({'gIoU': 0, 'delta': 0, 'ok': False, 'reason': 'no init bbox'})
        continue

    sb = 1.0 / sc
    b0 = [p0['bbox'][0]*sb, p0['bbox'][1]*sb, p0['bbox'][2]*sb, p0['bbox'][3]*sb]
    pts0 = [(p[0]*sb, p[1]*sb) for p in p0['points']]
    m0, sc0 = sam2_seg(inp, b0, pts0)
    g0 = giou(m0, gt)

    ys, xs = np.where(m0) if m0.sum() > 0 else ([0], [0])
    met = {'area': int(m0.sum()), 'coverage': m0.sum()/(m0.shape[0]*m0.shape[1]),
           'cx': xs.mean(), 'cy': ys.mean()}

    # Hybrid verification
    vis = make_outline_overlay(img, m0, b0, pts0, (wi, hi))
    vp = HYBRID_PROMPT.format(query=q, area=met['area'], coverage=met['coverage'],
                               cx=met['cx'], cy=met['cy'], score=sc0)
    out1 = qwen_chat(vis, vp, 300)
    p1 = parse_geometric(out1)

    g1 = g0
    if p1['bbox']:
        b1 = [p1['bbox'][0]*sb, p1['bbox'][1]*sb, p1['bbox'][2]*sb, p1['bbox'][3]*sb]
        pts1 = [(p[0]*sb, p[1]*sb) for p in p1['points']]
        m1, _ = sam2_seg(inp, b1, pts1)
        g1 = giou(m1, gt)

    improved = g1 > g0 + 0.01
    print(f"[{s['idx']:3d}] {q[:55]:55s} | {g0:.4f} -> {g1:.4f} "
          f"({'+' if improved else ' '}{g1-g0:+.4f})  parse={'OK' if p1['bbox'] else 'FAIL'}")

    results.append({'gIoU': g1, 'delta': g1-g0, 'ok': p1['bbox'] is not None})

    if (len(results)) % 50 == 0:
        ok_n = sum(1 for r in results if r['ok'])
        gs = [r['gIoU'] for r in results]
        print(f"  --- [{len(results)}/779] parse={ok_n}/{len(results)}, "
              f"mean_gIoU={np.mean(gs):.4f} ---")

n_samples = len(results)
ok = sum(1 for r in results if r['ok'])
gious = [r['gIoU'] for r in results]
deltas = [r['delta'] for r in results]
imp = sum(1 for d in deltas if d > 0.01)

print(f"\n{'='*60}")
print(f"  HYBRID Verification (outline + text metrics)")
print(f"{'='*60}")
print(f"  Samples:      {n_samples}")
print(f"  Bbox parse:   {ok}/{n_samples}")
print(f"  Mean gIoU:    {np.mean(gious):.4f}")
print(f"  Mean delta:   {np.mean(deltas):+.4f}")
print(f"  Improved:     {imp}/{n_samples}")
print(f"\n  vs. Pure Text baseline: 0.5025 (full 779)")

# Save
def to_native(v):
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    return v

save_data = [{'gIoU': float(r['gIoU']), 'delta': float(r['delta']),
              'ok': bool(r['ok'])} for r in results]
with open(Path(__file__).parent / 'reasonseg_output' / 'hybrid_results.json', 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"  Results saved: reasonseg_output/hybrid_results.json")
