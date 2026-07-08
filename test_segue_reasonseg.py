"""
test_segue_reasonseg.py — SEGUE on ReasonSeg 快速测试
"""
import sys, os, re, time
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, r'D:\afss')

import torch, numpy as np
from PIL import Image
from pathlib import Path

# ==== Load models ====
# Import transformers first
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

torch.cuda.empty_cache()
print(f"[1/4] Loading Qwen2.5-VL-3B... GPU free: {torch.cuda.mem_get_info()[0]/1e9:.2f}GB")
import sys  # make sure flush works
sys.stdout.flush()

lvlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype=torch.float16,
    device_map='auto', attn_implementation='sdpa', local_files_only=True,
).eval()
proc = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', local_files_only=True)
print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
sys.stdout.flush()

# Defer SAM2 import until after Qwen is safely loaded
print("[2/4] Loading SAM2 (CPU)...")
sys.stdout.flush()
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sam2_model = build_sam2(
    config_file='configs/sam2.1/sam2.1_hiera_b+.yaml',
    ckpt_path=r'D:\pycharm_projects\NOTRAING\NOTRAING\checkpoints\sam2.1_hiera_base_plus.pt',
    device='cpu',
)
sam2_pred = SAM2ImagePredictor(sam2_model)
print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
sys.stdout.flush()

print("[3/4] Loading ReasonSeg dataset...")
sys.stdout.flush()
from datasets import load_dataset
ds = load_dataset('Ricky06662/ReasonSeg_test')['test']
print(f"  {len(ds)} test samples")
sys.stdout.flush()

# ==== Test on first N samples ====
N_SAMPLES = 3
from qwen_vl_utils import process_vision_info

SEGUE_PROMPT = """You are a visual reasoning agent for precise segmentation. Given an image and query, output:

<REASON>
[Your step-by-step reasoning about the target's location, appearance, and distinguishing features]
</REASON>
<SEG_BOX>
x1 y1 x2 y2
</SEG_BOX>
<SEG_POINTS>
px1 py1
px2 py2
</SEG_POINTS>

The box should tightly enclose the target. Points should be definitely INSIDE the target.
Output coordinates in pixels. Image size will be provided."""

print(f"\n[4/4] Running SEGUE on {N_SAMPLES} samples...")
print("=" * 60)

results = []

for idx in range(N_SAMPLES):
    sample = ds[idx]
    img = sample['image'].convert('RGB')
    query = sample['text']
    gt_mask = np.array(sample['mask'], dtype=bool)
    w_orig, h_orig = img.size

    # Resize for Qwen
    scale = min(800 / max(w_orig, h_orig), 1.0)
    w_infer = int(w_orig * scale)
    h_infer = int(h_orig * scale)
    img_small = img.resize((w_infer, h_infer), Image.BILINEAR)

    print(f"\n--- Sample {idx}: {query[:80]}... ---")
    print(f"  Image: {w_orig}x{h_orig} -> {w_infer}x{h_infer}")

    # === LVLM Reasoning ===
    prompt = SEGUE_PROMPT + f"\n\nImage size: {w_infer} x {h_infer} pixels\n\nQuery: {query}\n\nNow generate:"
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
    print(f"  Qwen ({t_qwen:.1f}s):")
    print(f"    {out_text[:200].replace(chr(10), ' ')}...")

    # === Parse geometric prompts ===
    box_m = re.search(r'<SEG_BOX>\s*([\d\s\.]+)\s*</SEG_BOX>', out_text)
    pts_m = re.search(r'<SEG_POINTS>\s*([\d\s\.]+)\s*</SEG_POINTS>', out_text)

    if not box_m:
        print("  FAILED: no bbox parsed")
        results.append({'query': query, 'gIoU': 0, 'error': 'no bbox'})
        continue

    box_nums = [float(x) for x in re.findall(r'\d+\.?\d*', box_m.group(1))]
    if len(box_nums) < 4:
        print("  FAILED: incomplete bbox")
        results.append({'query': query, 'gIoU': 0, 'error': 'incomplete bbox'})
        continue

    bbox = [box_nums[0], box_nums[1], box_nums[2], box_nums[3]]
    print(f"  Parsed bbox: {bbox}")

    points = []
    if pts_m:
        pt_nums = [float(x) for x in re.findall(r'\d+\.?\d*', pts_m.group(1))]
        points = [(pt_nums[i], pt_nums[i+1]) for i in range(0, len(pt_nums)-1, 2)][:2]

    # Map to original image coordinates
    scale_back = 1.0 / scale
    bbox_orig = [b * scale_back for b in bbox]
    pts_orig = [(p[0] * scale_back, p[1] * scale_back) for p in points]

    # === SAM2 segmentation ===
    img_np = np.array(img)
    sam2_pred.set_image(img_np)
    pt_arr = np.array(pts_orig, dtype=np.float32) if pts_orig else None
    pt_labels = np.ones(len(pts_orig), dtype=np.int32) if pts_orig else None
    box_arr = np.array(bbox_orig, dtype=np.float32)

    t1 = time.time()
    masks, scores, _ = sam2_pred.predict(
        point_coords=pt_arr, point_labels=pt_labels,
        box=box_arr, multimask_output=False,
    )
    pred_mask = masks[0].astype(bool)
    t_sam2 = time.time() - t1

    # === Evaluate ===
    # Resize GT to match pred
    if gt_mask.shape != pred_mask.shape:
        gt_resized = np.array(Image.fromarray(gt_mask).resize(
            (pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST))
    else:
        gt_resized = gt_mask

    inter = (pred_mask & gt_resized).sum()
    union = (pred_mask | gt_resized).sum()
    giou = inter / max(union, 1)

    print(f"  SAM2 ({t_sam2:.1f}s): fg={pred_mask.sum()}px, score={scores[0]:.3f}")
    print(f"  gIoU: {giou:.4f}")

    results.append({
        'query': query,
        'gIoU': giou,
        'qwen_time': t_qwen,
        'sam2_time': t_sam2,
        'pred_area': int(pred_mask.sum()),
        'gt_area': int(gt_mask.sum()),
        'bbox': bbox_orig,
        'points': pts_orig,
        'output': out_text[:300],
    })

    # Save visualization
    if pred_mask.shape != (h_orig, w_orig):
        pred_vis = np.array(Image.fromarray(pred_mask).resize((w_orig, h_orig), Image.NEAREST))
    else:
        pred_vis = pred_mask

    vis = np.array(img).copy()
    overlay = vis.copy()
    overlay[gt_mask] = [0, 255, 0]       # GT = green
    overlay[pred_vis] = [255, 0, 0]       # Pred = red
    vis = (vis * 0.5 + overlay * 0.5).astype(np.uint8)

    out_dir = Path(__file__).parent / 'reasonseg_output'
    out_dir.mkdir(exist_ok=True)
    Image.fromarray(vis).save(out_dir / f'sample_{idx:02d}.jpg')

# ==== Summary ====
print(f"\n{'='*60}")
print(f"  SEGUE Zero-Shot on ReasonSeg (n={len(results)})")
print(f"{'='*60}")
gious = [r['gIoU'] for r in results]
print(f"  gIoU per sample: {[f'{g:.4f}' for g in gious]}")
print(f"  Mean gIoU: {np.mean(gious):.4f}")
print(f"\nVisualizations saved to: reasonseg_output/")
