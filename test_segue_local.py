"""
test_segue_local.py — SEGUE on ReasonSeg (本地文件版, 无datasets依赖)
"""
import sys, os, re, time, json
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'D:\afss')

import torch, numpy as np
from PIL import Image
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'reasonseg_data'

# ==== Load models ====
print(f"[1/3] Loading Qwen2.5-VL-3B... GPU free: {torch.cuda.mem_get_info()[0]/1e9:.2f}GB")
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
lvlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype=torch.float16,
    device_map='auto', attn_implementation='sdpa', local_files_only=True,
).eval()
proc = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', local_files_only=True)
print(f"  Qwen loaded. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

print(f"[2/3] Loading SAM2 (CPU)...")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sam2_model = build_sam2(
    config_file='configs/sam2.1/sam2.1_hiera_b+.yaml',
    ckpt_path=r'D:\pycharm_projects\NOTRAING\NOTRAING\checkpoints\sam2.1_hiera_base_plus.pt',
    device='cpu',
)
sam2_pred = SAM2ImagePredictor(sam2_model)
print(f"  SAM2 loaded. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

# ==== Load local data ====
print(f"[3/3] Loading local ReasonSeg data...")
samples = []
for i in range(20):
    img_path = DATA_DIR / f'{i:04d}.jpg'
    mask_path = DATA_DIR / f'{i:04d}_mask.npy'
    meta_path = DATA_DIR / f'{i:04d}.json'
    if img_path.exists() and mask_path.exists():
        samples.append({
            'image': Image.open(img_path).convert('RGB'),
            'mask': np.load(mask_path),
            'meta': json.load(open(meta_path)),
        })
print(f"  {len(samples)} samples loaded")

# ==== SEGUE Prompt ====
SEGUE_PROMPT = """You are a visual reasoning agent for precise segmentation. Given an image and query, output:

<REASON>
[Step-by-step reasoning about the target's visual appearance, location, and distinguishing features]
</REASON>
<SEG_BOX>
x1 y1 x2 y2
</SEG_BOX>
<SEG_POINTS>
px1 py1
px2 py2
</SEG_POINTS>

Rules:
- Box: tight bounding box in pixels (top-left x y, bottom-right x y)
- Points: two (x, y) that are definitely INSIDE the target
- All coordinates relative to the provided image dimensions"""

from qwen_vl_utils import process_vision_info

# ==== Test ====
N = min(5, len(samples))
print(f"\n{'='*60}")
print(f"  SEGUE on {N} ReasonSeg samples")
print(f"{'='*60}")

results = []
for idx in range(N):
    sample = samples[idx]
    img = sample['image']
    gt_mask = sample['mask']
    query = sample['meta']['text']
    w_orig, h_orig = img.size

    # Resize
    max_size = 800
    scale = min(max_size / max(w_orig, h_orig), 1.0)
    w_infer = max(64, int(w_orig * scale))
    h_infer = max(64, int(h_orig * scale))
    img_small = img.resize((w_infer, h_infer), Image.BILINEAR)

    print(f"\n--- [{idx}] {query[:80]}... ---")
    print(f"  {w_orig}x{h_orig} -> {w_infer}x{h_infer}")

    # === Qwen Reasoning ===
    prompt = SEGUE_PROMPT + f"\n\nImage size: {w_infer} x {h_infer} pixels\nQuery: {query}\nNow generate:"
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img_small},
        {'type': 'text', 'text': prompt},
    ]}]

    t0 = time.time()
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                  padding=True, return_tensors='pt').to(lvlm.device)

    with torch.no_grad():
        out_ids = lvlm.generate(**inputs, max_new_tokens=256, do_sample=False)
    out_text = proc.batch_decode(out_ids, skip_special_tokens=True)[0]
    if 'assistant' in out_text:
        out_text = out_text.split('assistant')[-1].strip()
    t_qwen = time.time() - t0

    # Show output
    print(f"  Qwen ({t_qwen:.1f}s):")
    for line in out_text.split('\n')[:5]:
        print(f"    {line[:120]}")

    # === Parse (normalize commas → spaces) ===
    out_text_clean = out_text.replace(',', ' ')
    box_m = re.search(r'<SEG_BOX>\s*([\d\s\.]+)\s*</SEG_BOX>', out_text_clean)
    pts_m = re.search(r'<SEG_POINTS>\s*([\d\s\.]+)\s*</SEG_POINTS>', out_text_clean)

    if not box_m:
        print(f"  FAIL: no bbox. Output: {out_text[:200]}")
        results.append({'gIoU': 0, 'error': 'no bbox'})
        continue

    nums = [float(x) for x in re.findall(r'\d+\.?\d*', box_m.group(1))]
    if len(nums) < 4:
        print(f"  FAIL: incomplete bbox: {nums}")
        results.append({'gIoU': 0, 'error': 'incomplete bbox'})
        continue

    bbox = nums[:4]
    points = []
    if pts_m:
        pnums = [float(x) for x in re.findall(r'\d+\.?\d*', pts_m.group(1))]
        points = [(pnums[i], pnums[i+1]) for i in range(0, len(pnums)-1, 2)][:2]

    # Scale back to original
    s_back = 1.0 / scale
    bbox_orig = [bbox[0]*s_back, bbox[1]*s_back, bbox[2]*s_back, bbox[3]*s_back]
    pts_orig = [(p[0]*s_back, p[1]*s_back) for p in points]

    # === SAM2 ===
    img_np = np.array(img)
    sam2_pred.set_image(img_np)
    pt_arr = np.array(pts_orig, dtype=np.float32) if pts_orig else None
    pt_labels = np.ones(len(pts_orig), dtype=np.int32) if pts_orig else None
    box_arr = np.array(bbox_orig, dtype=np.float32)

    t1 = time.time()
    masks, scores, _ = sam2_pred.predict(
        point_coords=pt_arr, point_labels=pt_labels, box=box_arr, multimask_output=False)
    pred_mask = masks[0].astype(bool)
    t_sam2 = time.time() - t1

    # === gIoU ===
    if gt_mask.shape != pred_mask.shape:
        gt_rs = np.array(Image.fromarray(gt_mask).resize(
            (pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST))
    else:
        gt_rs = gt_mask

    inter = (pred_mask & gt_rs).sum()
    union = (pred_mask | gt_rs).sum()
    giou = inter / max(union, 1)

    print(f"  SAM2 ({t_sam2:.1f}s): mask={pred_mask.sum()}px, score={scores[0]:.3f}")
    print(f"  gIoU: {giou:.4f}")

    results.append({'gIoU': giou, 'qwen_t': t_qwen, 'sam2_t': t_sam2,
                    'pred_area': int(pred_mask.sum()), 'gt_area': int(gt_mask.sum())})

    # Vis
    out_dir = Path(__file__).parent / 'reasonseg_output'
    out_dir.mkdir(exist_ok=True)
    if pred_mask.shape != (h_orig, w_orig):
        pred_vis = np.array(Image.fromarray(pred_mask).resize((w_orig, h_orig), Image.NEAREST))
    else:
        pred_vis = pred_mask

    vis = np.array(img).copy()[:,:,:3]  # ensure 3-channel
    overlay = vis.copy()
    overlay[gt_mask] = [0, 255, 0]
    overlay[pred_vis] = [255, 0, 0]
    blended = (vis * 0.5 + overlay * 0.5).astype(np.uint8)
    Image.fromarray(blended).save(out_dir / f'reasonseg_{idx:02d}.jpg')

# ==== Summary ====
gious = [r['gIoU'] for r in results]
print(f"\n{'='*60}")
print(f"  SEGUE Zero-Shot Results on ReasonSeg")
print(f"{'='*60}")
for i, r in enumerate(results):
    print(f"  [{i}] gIoU={r['gIoU']:.4f}")
print(f"  Mean gIoU: {np.mean(gious):.4f}")
print(f"  > 0.3: {sum(1 for g in gious if g > 0.3)}/{len(gious)}")
print(f"  Output: reasonseg_output/")
