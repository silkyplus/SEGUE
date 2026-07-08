"""
stage1_train.py — SEGUE Stage 1 训练脚本
=========================================
Grounding Pretrain: 训练 Prompt Refiner + LoRA on GDINO + 冻结 SAM2

用法:
    python stage1_train.py                    # 从头训练
    python stage1_train.py --resume <path>   # 从 checkpoint 恢复
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from stage1_config import *
from stage1_model import SEGUEStage1
from stage1_losses import GroundingLoss, masks_to_bboxes_and_points
from stage1_dataset import SatelliteGroundingDataset, SatelliteGroundingValDataset, grounding_collate_fn


# ============================================================
# 训练工具
# ============================================================
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_optimizer(model):
    """只优化可训练参数"""
    trainable_params = [
        p for p in model.parameters() if p.requires_grad
    ]

    # 分组: Prompt Refiner 和其他 (LoRA)
    refiner_params = []
    lora_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'prompt_refiner' in name:
                refiner_params.append(p)
            else:
                lora_params.append(p)

    param_groups = []
    if refiner_params:
        param_groups.append({
            'params': refiner_params,
            'lr': LR,
            'name': 'refiner',
        })
    if lora_params:
        param_groups.append({
            'params': lora_params,
            'lr': LR * 0.5,  # LoRA 用较低学习率
            'name': 'lora',
        })

    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    return optimizer


def save_checkpoint(model, optimizer, epoch, loss, path, is_best=False):
    """保存 checkpoint"""
    save_dict = {
        'epoch': epoch,
        'model_state_dict': {
            k: v for k, v in model.state_dict().items()
            if 'sam2' not in k  # 不保存冻结的 SAM2
        },
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(save_dict, path)
    if is_best:
        best_path = path.parent / 'best_model.pt'
        torch.save(save_dict, best_path)
        print(f"  ✓ Best model saved to {best_path}")


def load_checkpoint(model, optimizer, path, device):
    """加载 checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model_state = model.state_dict()
    # 只加载 prompt_refiner 和非 SAM2 的参数
    for k, v in checkpoint['model_state_dict'].items():
        if k in model_state:
            model_state[k] = v
    model.load_state_dict(model_state, strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


# ============================================================
# 验证函数
# ============================================================
@torch.no_grad()
def validate(model, val_loader, loss_fn, device, epoch):
    """验证: 计算 mask IoU + loss"""
    model.eval()
    total_loss = AverageMeter()
    total_iou = AverageMeter()

    for batch_idx, batch in enumerate(val_loader):
        images = batch['image'].to(device)
        captions = batch['caption']

        # Forward
        outputs = model(images, captions)

        # 准备 targets
        targets = {
            'gt_bboxes': [bb.to(device) for bb in batch['gt_bboxes']],
            'gt_points': [pt.to(device) for pt in batch['gt_points']],
            'gt_masks': [m.to(device) for m in batch['gt_masks']],
        }

        # Loss
        loss, loss_dict = loss_fn(outputs, targets)
        total_loss.update(loss_dict['total'], images.size(0))

        # Mask IoU (简化计算)
        iou_batch = compute_batch_iou(outputs['pred_masks'], batch['gt_masks'], device)
        total_iou.update(iou_batch, images.size(0))

        if batch_idx % 10 == 0:
            print(f"  Val [{batch_idx}/{len(val_loader)}] "
                  f"Loss: {loss_dict['total']:.4f} | IoU: {iou_batch:.4f}")

    print(f"\n[Epoch {epoch}] Validation: Loss={total_loss.avg:.4f}, mIoU={total_iou.avg:.4f}")
    return total_loss.avg, total_iou.avg


def compute_batch_iou(pred_masks_list, gt_masks_list, device):
    """计算 batch 级别的平均 IoU"""
    ious = []
    for pred_masks, gt_masks in zip(pred_masks_list, gt_masks_list):
        if len(pred_masks) == 0 or len(gt_masks) == 0:
            ious.append(0.0)
            continue

        pred_masks = pred_masks.to(device)
        gt_masks = gt_masks.to(device)

        # 确保尺寸一致
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            pred_masks = F.interpolate(
                pred_masks.unsqueeze(0) if pred_masks.dim() == 3 else pred_masks,
                size=gt_masks.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0) if pred_masks.dim() == 3 else pred_masks

        # 贪心匹配
        pred_bin = (pred_masks.sigmoid() > 0.5).float()
        gt_bin = gt_masks.float()

        # 对每个 GT，找最佳 pred
        per_gt_ious = []
        gt_matched = set()
        for g in range(len(gt_bin)):
            best_iou = 0.0
            best_p = -1
            for p in range(len(pred_bin)):
                if p in gt_matched:
                    continue
                intersection = (pred_bin[p] * gt_bin[g]).sum()
                union = (pred_bin[p] + gt_bin[g]).clamp(0, 1).sum()
                iou = (intersection + 1e-6) / (union + 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_p = p
            if best_p >= 0:
                gt_matched.add(best_p)
                per_gt_ious.append(best_iou.item())

        ious.append(np.mean(per_gt_ious) if per_gt_ious else 0.0)

    return np.mean(ious)


# ============================================================
# 训练主循环
# ============================================================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    set_seed()

    # ---- 创建输出目录 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"stage1_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # ---- 模型 (先加载，避免多进程冲突) ----
    print("\n" + "=" * 60)
    print("  Building model...")
    print("=" * 60)
    model = SEGUEStage1(device=device)
    print("  Model loaded. Now loading datasets...")

    # ---- 数据集 ----
    train_dataset = SatelliteGroundingDataset(is_train=True)
    val_dataset = SatelliteGroundingValDataset()

    # Windows下 num_workers=0 避免多进程问题
    safe_workers = 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=safe_workers,
        collate_fn=grounding_collate_fn,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=safe_workers,
        collate_fn=grounding_collate_fn,
        pin_memory=False,
    )
    print(f"  Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params:    {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # ---- 优化器 & 损失 ----
    optimizer = get_optimizer(model)
    loss_fn = GroundingLoss()

    # ---- 学习率调度 ----
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,      # 初始周期 10 epochs
        T_mult=2,    # 每个周期乘 2
        eta_min=LR * 0.01,
    )

    # ---- Resume ----
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch, best_loss = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1

    # ---- 训练循环 ----
    print("\n" + "=" * 60)
    print("  Starting training...")
    print("=" * 60)
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    print(f"  Train steps/epoch: {len(train_loader)}")
    print("=" * 60)

    patience_counter = 0

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = AverageMeter()
        epoch_bbox_loss = AverageMeter()
        epoch_point_loss = AverageMeter()
        epoch_mask_loss = AverageMeter()
        epoch_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            captions = batch['caption']

            # ---- Forward ----
            outputs = model(images, captions)

            # ---- 准备 GT targets ----
            targets = {
                'gt_bboxes': [bb.to(device) for bb in batch['gt_bboxes']],
                'gt_points': [pt.to(device) for pt in batch['gt_points']],
                'gt_masks': [m.to(device) for m in batch['gt_masks']],
            }

            # ---- Loss ----
            loss, loss_dict = loss_fn(outputs, targets)

            if torch.isnan(loss) or loss.item() == 0.0:
                continue  # skip empty batches

            # ---- Backward ----
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                GRAD_CLIP,
            )
            optimizer.step()

            # ---- Logging ----
            epoch_loss.update(loss_dict['total'])
            epoch_bbox_loss.update(loss_dict['bbox'])
            epoch_point_loss.update(loss_dict['point'])
            epoch_mask_loss.update(loss_dict['mask'])

            if batch_idx % LOG_EVERY == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"  [E{epoch:02d} | {batch_idx:04d}/{len(train_loader)}] "
                      f"L={loss_dict['total']:.4f} "
                      f"bbox={loss_dict['bbox']:.4f} "
                      f"pt={loss_dict['point']:.4f} "
                      f"mask={loss_dict['mask']:.4f} "
                      f"lr={lr:.6f}")

        # ---- Epoch 结束 ----
        epoch_time = time.time() - epoch_time
        print(f"\n[Epoch {epoch:02d}] Time: {epoch_time:.1f}s | "
              f"Loss: {epoch_loss.avg:.4f} | "
              f"bbox: {epoch_bbox_loss.avg:.4f} | "
              f"pt: {epoch_point_loss.avg:.4f} | "
              f"mask: {epoch_mask_loss.avg:.4f}")

        # ---- 学习率更新 ----
        scheduler.step()

        # ---- 验证 ----
        if epoch % VAL_EVERY == 0:
            val_loss, val_iou = validate(model, val_loader, loss_fn, device, epoch)

            # ---- Early stopping ----
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch, val_loss,
                                run_dir / f'checkpoint_epoch_{epoch:02d}.pt',
                                is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\n  Early stopping at epoch {epoch}")
                    break

            # 常规保存
            save_checkpoint(model, optimizer, epoch, val_loss,
                            run_dir / f'checkpoint_epoch_{epoch:02d}.pt')

    # ---- 训练结束 ----
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Results saved to: {run_dir}")
    print("=" * 60)

    # 保存最终配置
    config_save = {k: str(v) if isinstance(v, Path) else v
                   for k, v in globals().items()
                   if k.isupper() and not k.startswith('_')}
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_save, f, indent=2, default=str)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SEGUE Stage 1 Training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=LR,
                        help='Learning rate')
    args = parser.parse_args()

    # Override config
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr

    train(args)
