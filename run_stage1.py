"""
SEGUE Stage 1 快速训练脚本 (L_geom only, no SAM2)
直接运行: python run_stage1.py
"""
import sys, os, time
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'D:\afss')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress verbose warnings
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from stage1_config import *
from stage1_model import SEGUEStage1
from stage1_losses import GroundingLoss
from stage1_dataset import SatelliteGroundingDataset, SatelliteGroundingValDataset, grounding_collate_fn

def main():
    print("="*60)
    print("  SEGUE Stage 1 — Grounding Pretrain (L_geom)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED)

    # ---- Data ----
    print("\nLoading datasets...")
    train_ds = SatelliteGroundingDataset(is_train=True)
    val_ds = SatelliteGroundingValDataset()
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=grounding_collate_fn, num_workers=0)
    print(f"  Train: {len(train_ds)} samples, {len(train_loader)} batches")

    # ---- Model ----
    print("\nLoading model (GDINO + Prompt Refiner + SAM2)...")
    sys.stdout.flush()
    model = SEGUEStage1(device=device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"  GDINO LoRA: {sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and 'gdino' in n.lower()):,}")
    print(f"  Refiner:    {sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and 'refiner' in n.lower()):,}")

    # ---- Optimizer & Loss ----
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = GroundingLoss()

    # ---- Output ----
    run_dir = Path(__file__).parent / 'runs' / f'stage1_{time.strftime("%Y%m%d_%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun dir: {run_dir}")

    # ---- Training Loop ----
    N_EPOCHS = 10
    print(f"\n{'='*60}")
    print(f"  Training {N_EPOCHS} epochs (no SAM2, fast mode)")
    print(f"{'='*60}")

    for epoch in range(N_EPOCHS):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        epoch_bbox = 0.0
        epoch_point = 0.0
        n_batches = 0

        for bi, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            captions = batch['caption']

            # Forward (no SAM2 for speed)
            outputs = model(images, captions, run_sam2=False)

            targets = {
                'gt_bboxes': [bb.to(device) for bb in batch['gt_bboxes']],
                'gt_points': [pt.to(device) for pt in batch['gt_points']],
                'gt_masks': [m.to(device) for m in batch['gt_masks']],
            }

            loss, d = loss_fn(outputs, targets, compute_mask_loss=False)

            if loss.item() == 0:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()

            epoch_loss += d['total']
            epoch_bbox += d['bbox']
            epoch_point += d['point']
            n_batches += 1

            if bi % 200 == 0:
                print(f"  [E{epoch:02d}|{bi:04d}/{len(train_loader)}] "
                      f"L={d['total']:.2f} bbox={d['bbox']:.3f} pt={d['point']:.1f}")

        t1 = time.time()
        n = max(1, n_batches)
        print(f"[Epoch {epoch:02d}] Loss: {epoch_loss/n:.2f}  "
              f"bbox: {epoch_bbox/n:.3f}  pt: {epoch_point/n:.1f}  "
              f"Time: {t1-t0:.1f}s")

        # Save
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            ckpt_path = run_dir / f'ckpt_e{epoch:02d}.pt'
            torch.save({
                'epoch': epoch,
                'model': {k: v for k, v in model.state_dict().items() if 'sam2' not in k},
                'optimizer': optimizer.state_dict(),
                'loss': epoch_loss / n,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    print(f"\nDone! Checkpoints: {run_dir}")

if __name__ == '__main__':
    main()
