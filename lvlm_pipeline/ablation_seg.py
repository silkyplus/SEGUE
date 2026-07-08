"""
ablation_seg_scratch.py  —  全监督语义分割消融实验（无预训练版）
mIoU 同时输出:
  - mIoU(fg)  : 前景3类均值 (body / solar_panel / antenna)
  - mIoU(all) : 含背景4类均值
依赖安装:
    pip install segmentation-models-pytorch matplotlib
"""

import os, sys, json, csv, time, traceback, random, heapq
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# ==================== 全局配置 ====================
CFG = {
    "train_img_dir":  r"D:\data\final_dataset\final_dataset\images\train",
    "train_mask_dir": r"D:\data\final_dataset\final_dataset\masks\train",
    "val_img_dir":    r"D:\data\final_dataset\final_dataset\images\val",
    "val_mask_dir":   r"D:\data\final_dataset\final_dataset\masks\val",

    "exp_root": r"D:\data\final_dataset\experiments_scratch",

    "num_classes": 4,
    "image_size":  (512, 512),
    "epochs":              100,
    "batch_size":          4,
    "num_workers":         0,
    "lr":                  1e-3,
    "weight_decay":        1e-4,
    "early_stop_patience": 20,
    "device":              "cuda",
    "seed":                42,
    "top_k_vis":           10,
}

MASK_COLOR_TO_ID = {
    (0,   0,   0): 0,
    (0, 255,   0): 1,   # body
    (0,   0, 255): 2,   # solar_panel
    (255,  0,   0): 3,  # antenna
}
CLASS_COLORS_BGR = {
    0: (0,   0,   0),
    1: (0, 255,   0),
    2: (0,   0, 255),
    3: (255,  0,   0),
}
CLASS_NAMES = ["background", "body", "solar_panel", "antenna"]

# ==================== 消融实验组 ====================
EXPERIMENTS = [
    # ---------- 已有实验（已完成则自动跳过）----------
    {"name": "unet_resnet18_scratch",       "model_arch": "unet",          "encoder": "resnet18"},
    {"name": "unet_resnet34_scratch",       "model_arch": "unet",          "encoder": "resnet34"},
    {"name": "unet_resnet50_scratch",       "model_arch": "unet",          "encoder": "resnet50"},
    {"name": "unetpp_resnet18_scratch",     "model_arch": "unetplusplus",  "encoder": "resnet18"},
    {"name": "unetpp_resnet34_scratch",     "model_arch": "unetplusplus",  "encoder": "resnet34"},
    {"name": "deeplabv3p_resnet18_scratch", "model_arch": "deeplabv3plus", "encoder": "resnet18"},
    {"name": "deeplabv3p_resnet34_scratch", "model_arch": "deeplabv3plus", "encoder": "resnet34"},
    {"name": "deeplabv3p_resnet50_scratch", "model_arch": "deeplabv3plus", "encoder": "resnet50"},
    {"name": "fpn_resnet18_scratch",        "model_arch": "fpn",           "encoder": "resnet18"},
    {"name": "fpn_resnet34_scratch",        "model_arch": "fpn",           "encoder": "resnet34"},
    {"name": "pspnet_resnet18_scratch",     "model_arch": "pspnet",        "encoder": "resnet18"},
    {"name": "pspnet_resnet34_scratch",     "model_arch": "pspnet",        "encoder": "resnet34"},
    {"name": "manet_resnet18_scratch",      "model_arch": "manet",         "encoder": "resnet18"},
    {"name": "manet_resnet34_scratch",      "model_arch": "manet",         "encoder": "resnet34"},
    {"name": "pan_resnet18_scratch",        "model_arch": "pan",           "encoder": "resnet18"},
    {"name": "pan_resnet34_scratch",        "model_arch": "pan",           "encoder": "resnet34"},

    # ---------- 新增：LinkNet（轻量解码器，速度快）----------
    # LinkNet 用逐层转置卷积做解码，参数量远小于 U-Net
    {"name": "linknet_resnet18_scratch",    "model_arch": "linknet",       "encoder": "resnet18"},
    {"name": "linknet_resnet34_scratch",    "model_arch": "linknet",       "encoder": "resnet34"},
    {"name": "linknet_resnet50_scratch",    "model_arch": "linknet",       "encoder": "resnet50"},

    # ---------- 新增：DeepLabV3（标准版，不含 ASPP+解码器）----------
    {"name": "deeplabv3_resnet18_scratch",  "model_arch": "deeplabv3",     "encoder": "resnet18"},
    {"name": "deeplabv3_resnet34_scratch",  "model_arch": "deeplabv3",     "encoder": "resnet34"},
    {"name": "deeplabv3_resnet50_scratch",  "model_arch": "deeplabv3",     "encoder": "resnet50"},

    # ---------- 新增：UNet++ resnet50（原列表缺失）----------
    {"name": "unetpp_resnet50_scratch",     "model_arch": "unetplusplus",  "encoder": "resnet50"},


]


# ==================== 数据集 ====================

def mask_to_label(mask_bgr: np.ndarray) -> np.ndarray:
    label = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    for (b, g, r), cls_id in MASK_COLOR_TO_ID.items():
        if cls_id == 0:
            continue
        label[np.all(mask_bgr == np.array([b, g, r]), axis=2)] = cls_id
    return label


def label_to_color(label: np.ndarray) -> np.ndarray:
    color_img = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS_BGR.items():
        color_img[label == cls_id] = color
    return color_img


class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_size, augment=False):
        self.img_paths  = sorted(Path(img_dir).glob("*.png"))
        self.mask_dir   = Path(mask_dir)
        self.image_size = image_size
        self.augment    = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        mask_path = self.mask_dir / f"{img_path.stem}_mask.png"

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        label = mask_to_label(cv2.imread(str(mask_path))) if mask_path.exists() \
                else np.zeros(image.shape[:2], dtype=np.uint8)

        W, H  = self.image_size
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, label = self._augment(image, label)

        image = (image.astype(np.float32) / 255.0
                 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return (torch.from_numpy(image.transpose(2, 0, 1)).float(),
                torch.from_numpy(label).long(),
                str(img_path))

    def _augment(self, image, label):
        if random.random() > 0.5:
            image = cv2.flip(image, 1); label = cv2.flip(label, 1)
        if random.random() > 0.5:
            image = cv2.flip(image, 0); label = cv2.flip(label, 0)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k).copy(); label = np.rot90(label, k).copy()
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            image = np.clip(image.astype(np.float32) * alpha
                            + random.randint(-20, 20), 0, 255).astype(np.uint8)
        return image, label


# ==================== 损失函数 ====================

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        pred_soft = torch.softmax(pred, dim=1)
        loss = 0.0
        for c in range(self.num_classes):
            pred_c = pred_soft[:, c]
            tgt_c  = (target == c).float()
            inter  = (pred_c * tgt_c).sum()
            loss  += 1 - (2 * inter + self.smooth) / (
                pred_c.sum() + tgt_c.sum() + self.smooth)
        return loss / self.num_classes


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.alpha = alpha

    def forward(self, pred, target):
        return (self.alpha * self.ce(pred, target)
                + (1 - self.alpha) * self.dice(pred, target))


# ==================== 模型构建（无预训练）====================

def build_model(arch, encoder, num_classes):
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("请先安装: pip install segmentation-models-pytorch")
    arch_map = {
        "unet":          smp.Unet,
        "unetplusplus":  smp.UnetPlusPlus,
        "deeplabv3":     smp.DeepLabV3,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn":           smp.FPN,
        "pspnet":        smp.PSPNet,
        "pan":           smp.PAN,
        "linknet":       smp.Linknet,
        "manet":         smp.MAnet,
    }
    if arch not in arch_map:
        raise ValueError(f"不支持的架构: {arch}，可用: {list(arch_map.keys())}")
    # PSPNet 要求输入尺寸可被 6*downsample_factor 整除，512 满足条件，无需特殊处理
    # EfficientNet encoder 名称: efficientnet-b0 ~ efficientnet-b7
    # MobileNet encoder 名称: mobilenet_v2
    return arch_map[arch](
        encoder_name=encoder,
        encoder_weights=None,   # 从零初始化，不下载预训练权重    encoder_weights = "imagenet" 则加载预训练权重
        in_channels=3,
        classes=num_classes,
    )


# ==================== 指标计算 ====================

class SegMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred   = pred.argmax(dim=1).cpu().numpy().ravel()
        target = target.cpu().numpy().ravel()
        valid  = target < self.num_classes
        np.add.at(self.confusion.ravel(),
                  target[valid] * self.num_classes + pred[valid], 1)

    def iou_per_class(self):
        """返回每个类别的 IoU，共 num_classes 个"""
        ious = []
        for c in range(self.num_classes):
            tp    = self.confusion[c, c]
            denom = self.confusion[c].sum() + self.confusion[:, c].sum() - tp
            ious.append(float(tp) / float(denom) if denom > 0 else float("nan"))
        return ious

    def miou_fg(self):
        """前景3类 mIoU，跳过 class 0（背景）"""
        ious  = self.iou_per_class()
        valid = [v for v in ious[1:] if not np.isnan(v)]
        return float(np.mean(valid)) if valid else 0.0

    def miou_all(self):
        """含背景的4类 mIoU"""
        valid = [v for v in self.iou_per_class() if not np.isnan(v)]
        return float(np.mean(valid)) if valid else 0.0

    def pixel_acc(self):
        total = self.confusion.sum()
        return float(np.diag(self.confusion).sum()) / float(total) if total > 0 else 0.0

    def summary(self):
        """返回所有指标字典"""
        ious = self.iou_per_class()
        def safe(v): return round(v, 4) if not np.isnan(v) else float("nan")
        return {
            "miou_fg":         round(self.miou_fg(),  4),
            "miou_all":        round(self.miou_all(), 4),
            "iou_background":  safe(ious[0]),
            "iou_body":        safe(ious[1]),
            "iou_solar_panel": safe(ious[2]),
            "iou_antenna":     safe(ious[3]),
            "pixel_acc":       round(self.pixel_acc(), 4),
        }


# ==================== 可视化：Loss 曲线（3图）====================

def save_loss_curve(log_path: Path, exp_dir: Path, exp_name: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [跳过曲线图] 请安装: pip install matplotlib")
        return

    epochs, train_losses, val_losses = [], [], []
    miou_fgs, miou_alls = [], []

    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row["train_loss"]))
                val_losses.append(float(row["val_loss"]))
                miou_fgs.append(float(row["miou_fg"]))
                miou_alls.append(float(row["miou_all"]))
            except (ValueError, KeyError):
                continue

    if not epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f"{exp_name}  —  Training Curves (no pretrain)",
                 fontsize=13, fontweight="bold")

    # 图1: Loss
    axes[0].plot(epochs, train_losses, label="Train Loss", color="#4C72B0", linewidth=1.5)
    axes[0].plot(epochs, val_losses,   label="Val Loss",   color="#DD8452", linewidth=1.5)
    best_ep = epochs[int(np.argmin(val_losses))]
    axes[0].axvline(x=best_ep, color="gray", linestyle="--", alpha=0.5,
                    label=f"Best ep={best_ep}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # 图2: mIoU(fg) 前景3类
    axes[1].plot(epochs, miou_fgs, label="mIoU(fg) 前景3类",
                 color="#55A868", linewidth=1.5)
    best_fg    = max(miou_fgs)
    best_fg_ep = epochs[int(np.argmax(miou_fgs))]
    axes[1].axvline(x=best_fg_ep, color="#55A868", linestyle="--", alpha=0.5)
    axes[1].annotate(f"Best={best_fg:.4f}\n(ep {best_fg_ep})",
                     xy=(best_fg_ep, best_fg),
                     xytext=(best_fg_ep + max(1, len(epochs)*0.05), best_fg - 0.04),
                     fontsize=9, color="#55A868",
                     arrowprops=dict(arrowstyle="->", color="#55A868", lw=1.2))
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU")
    axes[1].set_title("mIoU(fg) — 前景3类"); axes[1].grid(alpha=0.3); axes[1].legend()

    # 图3: mIoU(all) 含背景4类
    axes[2].plot(epochs, miou_alls, label="mIoU(all) 含背景4类",
                 color="#C44E52", linewidth=1.5)
    best_all    = max(miou_alls)
    best_all_ep = epochs[int(np.argmax(miou_alls))]
    axes[2].axvline(x=best_all_ep, color="#C44E52", linestyle="--", alpha=0.5)
    axes[2].annotate(f"Best={best_all:.4f}\n(ep {best_all_ep})",
                     xy=(best_all_ep, best_all),
                     xytext=(best_all_ep + max(1, len(epochs)*0.05), best_all - 0.04),
                     fontsize=9, color="#C44E52",
                     arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2))
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("mIoU")
    axes[2].set_title("mIoU(all) — 含背景4类"); axes[2].grid(alpha=0.3); axes[2].legend()

    plt.tight_layout()
    save_path = exp_dir / "loss_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Loss 曲线（3图）: {save_path}")


# ==================== 可视化：Top-K 效果图 ====================

def save_topk_vis(model, val_ds, device, exp_dir: Path,
                  num_classes: int, top_k: int = 10):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [跳过效果图] 请安装: pip install matplotlib")
        return

    vis_dir = exp_dir / "top_vis"
    vis_dir.mkdir(exist_ok=True)

    def collate_fn(batch):
        imgs   = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        paths  = [b[2] for b in batch]
        return imgs, labels, paths

    single_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                               num_workers=0, collate_fn=collate_fn)
    model.eval()
    heap = []

    with torch.no_grad():
        for idx, (images, labels, img_paths) in enumerate(
                tqdm(single_loader, desc="  Top-K 推理", leave=False)):
            images = images.to(device)
            outputs    = model(images)
            pred_label = outputs.argmax(dim=1).squeeze(0).cpu().numpy()
            gt_label   = labels.squeeze(0).cpu().numpy()

            metrics = SegMetrics(num_classes)
            metrics.update(outputs, labels.to(device))
            img_miou_fg = metrics.miou_fg()   # 排名用前景 mIoU

            orig  = cv2.imread(img_paths[0])
            entry = (img_miou_fg, idx, img_paths[0], pred_label, gt_label, orig)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif img_miou_fg > heap[0][0]:
                heapq.heapreplace(heap, entry)

    top_results = sorted(heap, key=lambda x: x[0], reverse=True)

    for rank, (img_miou_fg, idx, img_path, pred_label, gt_label, orig) in \
            enumerate(top_results, 1):

        stem = Path(img_path).stem
        W, H = CFG["image_size"]

        orig_resized = cv2.resize(orig, (W, H))
        orig_rgb     = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
        gt_rgb       = cv2.cvtColor(label_to_color(gt_label),   cv2.COLOR_BGR2RGB)
        pred_rgb     = cv2.cvtColor(label_to_color(pred_label), cv2.COLOR_BGR2RGB)

        overlay = orig_rgb.copy().astype(np.float32)
        for cls_id, (b, g, r) in CLASS_COLORS_BGR.items():
            if cls_id == 0:
                continue
            mask_region = pred_label == cls_id
            overlay[mask_region] = overlay[mask_region] * 0.55 + np.array([r, g, b]) * 0.45
        overlay = overlay.astype(np.uint8)
        for cls_id, (b, g, r) in CLASS_COLORS_BGR.items():
            if cls_id == 0:
                continue
            contours, _ = cv2.findContours(
                (pred_label == cls_id).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (r, g, b), 2)

        # 各类别 IoU（含背景）
        per_iou = {}
        for c in range(num_classes):
            pc = pred_label == c; gc = gt_label == c
            inter = (pc & gc).sum(); union = (pc | gc).sum()
            per_iou[CLASS_NAMES[c]] = inter / union if union > 0 else float("nan")

        fg_vals  = [v for k, v in per_iou.items() if k != "background" and not np.isnan(v)]
        all_vals = [v for v in per_iou.values() if not np.isnan(v)]
        cur_miou_fg  = np.mean(fg_vals)  if fg_vals  else 0.0
        cur_miou_all = np.mean(all_vals) if all_vals else 0.0

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle(
            f"Rank {rank}  |  {stem}\n"
            f"mIoU(fg)={cur_miou_fg:.4f}  mIoU(all)={cur_miou_all:.4f}  |  "
            f"bg={per_iou['background']:.3f}  body={per_iou['body']:.3f}  "
            f"solar={per_iou['solar_panel']:.3f}  ant={per_iou['antenna']:.3f}",
            fontsize=10, fontweight="bold"
        )
        for ax, (img, title) in zip(axes, [
            (orig_rgb, "Original"), (gt_rgb, "GT Mask"),
            (pred_rgb, "Pred Mask"), (overlay, "Overlay")
        ]):
            ax.imshow(img); ax.set_title(title, fontsize=10); ax.axis("off")

        from matplotlib.patches import Patch
        axes[-1].legend(handles=[
            Patch(facecolor=(0, 1, 0), label="body"),
            Patch(facecolor=(0, 0, 1), label="solar_panel"),
            Patch(facecolor=(1, 0, 0), label="antenna"),
        ], loc="lower right", fontsize=8, framealpha=0.8)

        plt.tight_layout()
        plt.savefig(vis_dir / f"top{rank:02d}_miou{cur_miou_fg:.4f}_{stem}.png",
                    dpi=120, bbox_inches="tight")
        plt.close()

    print(f"  ✓ Top-{top_k} 效果图: {vis_dir}")


# ==================== 训练单个实验 ====================

def run_experiment(exp_cfg: dict, global_cfg: dict):
    name     = exp_cfg["name"]
    exp_dir  = Path(global_cfg["exp_root"]) / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    done_flag = exp_dir / "done.json"
    ckpt_path = exp_dir / "checkpoint.pth"
    best_path = exp_dir / "best_model.pth"
    log_path  = exp_dir / "train_log.csv"

    if done_flag.exists():
        print(f"  [跳过] {name} 已完成")
        with open(done_flag) as f:
            return json.load(f)

    print(f"\n{'='*65}")
    print(f"  实验: {name}")
    print(f"  模型: {exp_cfg['model_arch']} + {exp_cfg['encoder']} (无预训练)")
    print(f"  损失: CE+Dice  |  LR: {global_cfg['lr']}  |  Epochs: {global_cfg['epochs']}")
    print(f"  输出: mIoU(fg)=前景3类  mIoU(all)=含背景4类  每epoch均输出")
    print(f"{'='*65}")

    device      = torch.device(global_cfg["device"] if torch.cuda.is_available() else "cpu")
    num_classes = global_cfg["num_classes"]
    epochs      = global_cfg["epochs"]

    seed = global_cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds = SegDataset(global_cfg["train_img_dir"], global_cfg["train_mask_dir"],
                          global_cfg["image_size"], augment=True)
    val_ds   = SegDataset(global_cfg["val_img_dir"],   global_cfg["val_mask_dir"],
                          global_cfg["image_size"], augment=False)

    def collate_fn(batch):
        imgs   = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        paths  = [b[2] for b in batch]
        return imgs, labels, paths

    train_loader = DataLoader(train_ds, batch_size=global_cfg["batch_size"],
                              shuffle=True,  num_workers=global_cfg["num_workers"],
                              pin_memory=True, drop_last=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=global_cfg["batch_size"],
                              shuffle=False, num_workers=global_cfg["num_workers"],
                              pin_memory=True, collate_fn=collate_fn)

    model     = build_model(exp_cfg["model_arch"], exp_cfg["encoder"], num_classes).to(device)
    criterion = CombinedLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=global_cfg["lr"],
                            weight_decay=global_cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0; best_miou_fg = 0.0; early_stop_cnt = 0; history = []

    if ckpt_path.exists():
        print(f"  恢复 checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch    = ckpt["epoch"] + 1
        best_miou_fg   = ckpt["best_miou_fg"]
        early_stop_cnt = ckpt.get("early_stop_cnt", 0)
        history        = ckpt.get("history", [])
        print(f"  从 epoch {start_epoch} 继续，最优 mIoU(fg)={best_miou_fg:.4f}")

    # CSV 表头：miou_fg 和 miou_all 双列
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss",
                "miou_fg",           # 前景3类 mIoU ← 主指标
                "miou_all",          # 含背景4类 mIoU ← 参考
                "iou_background",
                "iou_body",
                "iou_solar_panel",
                "iou_antenna",
                "pixel_acc", "lr", "time_s",
            ])

    # ===== 训练循环 =====
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # --- train ---
        model.train()
        train_loss = 0.0
        for images, labels, _ in tqdm(train_loader,
                                      desc=f"  Ep{epoch+1:03d}/{epochs} train",
                                      leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # --- val ---
        model.eval()
        val_loss = 0.0
        metrics  = SegMetrics(num_classes)
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader,
                                          desc=f"  Ep{epoch+1:03d}/{epochs} val",
                                          leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                metrics.update(outputs, labels)
        val_loss /= len(val_loader)

        # --- 同时计算两种 mIoU ---
        m            = metrics.summary()
        val_miou_fg  = m["miou_fg"]
        val_miou_all = m["miou_all"]
        cur_lr       = scheduler.get_last_lr()[0]
        elapsed      = time.time() - t0

        def fmt(v): return f"{v:.4f}" if not np.isnan(v) else "nan"

        # 写 CSV（含 miou_fg / miou_all 双列）
        row = [
            epoch + 1,
            f"{train_loss:.5f}", f"{val_loss:.5f}",
            f"{val_miou_fg:.5f}",
            f"{val_miou_all:.5f}",
            fmt(m["iou_background"]),
            fmt(m["iou_body"]),
            fmt(m["iou_solar_panel"]),
            fmt(m["iou_antenna"]),
            f"{m['pixel_acc']:.5f}",
            f"{cur_lr:.2e}", f"{elapsed:.1f}",
        ]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        history.append(row)

        # 终端打印：同时显示两种 mIoU + 各类别
        print(f"  Ep{epoch+1:3d} | "
              f"loss={train_loss:.4f} val={val_loss:.4f} | "
              f"mIoU(fg)={val_miou_fg:.4f}  mIoU(all)={val_miou_all:.4f} | "
              f"bg={fmt(m['iou_background'])} "
              f"body={fmt(m['iou_body'])} "
              f"solar={fmt(m['iou_solar_panel'])} "
              f"ant={fmt(m['iou_antenna'])} | "
              f"{elapsed:.0f}s")

        # 早停和保存以前景 mIoU 为准
        if val_miou_fg > best_miou_fg:
            best_miou_fg   = val_miou_fg
            early_stop_cnt = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ 新最优  mIoU(fg)={best_miou_fg:.4f}  "
                  f"mIoU(all)={val_miou_all:.4f}")
        else:
            early_stop_cnt += 1

        torch.save({
            "epoch":          epoch,
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "best_miou_fg":   best_miou_fg,
            "early_stop_cnt": early_stop_cnt,
            "history":        history,
        }, ckpt_path)

        if early_stop_cnt >= global_cfg["early_stop_patience"]:
            print(f"  早停触发（{global_cfg['early_stop_patience']} epoch 无提升）")
            break

    # ===== 最终评估 =====
    print(f"\n  最终评估（best_model）...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    metrics = SegMetrics(num_classes)
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc="  Final eval", leave=False):
            metrics.update(model(images.to(device)), labels.to(device))

    m = metrics.summary()

    # 终端完整打印最终结果
    print(f"\n  ===== 最终结果：{name} =====")
    print(f"  mIoU(fg)  [前景3类]  : {m['miou_fg']:.4f}   ← 主指标")
    print(f"  mIoU(all) [含背景4类]: {m['miou_all']:.4f}   ← 参考")
    print(f"  --------------------------------")
    print(f"  background  : {fmt(m['iou_background'])}")
    print(f"  body        : {fmt(m['iou_body'])}")
    print(f"  solar_panel : {fmt(m['iou_solar_panel'])}")
    print(f"  antenna     : {fmt(m['iou_antenna'])}")
    print(f"  pixel_acc   : {m['pixel_acc']:.4f}")

    result = {
        "name":            name,
        "model_arch":      exp_cfg["model_arch"],
        "encoder":         exp_cfg["encoder"],
        "pretrained":      False,
        "miou_fg":         m["miou_fg"],
        "miou_all":        m["miou_all"],
        "iou_background":  m["iou_background"],
        "iou_body":        m["iou_body"],
        "iou_solar_panel": m["iou_solar_panel"],
        "iou_antenna":     m["iou_antenna"],
        "pixel_acc":       m["pixel_acc"],
        "total_epochs":    len(history),
        "finished_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(done_flag, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    if ckpt_path.exists():
        ckpt_path.unlink()

    print(f"\n  绘制 Loss 曲线（3图）...")
    save_loss_curve(log_path, exp_dir, name)

    print(f"  生成 Top-{global_cfg['top_k_vis']} 效果图...")
    save_topk_vis(model, val_ds, device, exp_dir,
                  num_classes, top_k=global_cfg["top_k_vis"])

    return result


# ==================== 汇总打印 ====================

def print_summary(all_results: list):
    if not all_results:
        return
    all_results = sorted(all_results, key=lambda x: x.get("miou_fg", 0), reverse=True)

    print("\n" + "="*112)
    print("【消融实验汇总（按 mIoU(fg) 前景3类排序）】")
    print("="*112)
    print(f"{'#':>3} {'实验名称':<32} {'模型':<14} {'编码器':<10} "
          f"{'mIoU(fg)':>9} {'mIoU(all)':>10} "
          f"{'bg':>7} {'body':>7} {'solar':>7} {'antenna':>8} {'pix_acc':>8}")
    print("-"*112)
    for i, r in enumerate(all_results, 1):
        def s(k): return str(r.get(k, "?"))
        print(f"{i:>3} {r['name']:<32} {r['model_arch']:<14} {r['encoder']:<10} "
              f"{s('miou_fg'):>9} {s('miou_all'):>10} "
              f"{s('iou_background'):>7} {s('iou_body'):>7} "
              f"{s('iou_solar_panel'):>7} {s('iou_antenna'):>8} "
              f"{s('pixel_acc'):>8}")
    print("="*112)


def save_summary_csv(all_results: list, exp_root: str):
    if not all_results:
        return
    all_results = sorted(all_results, key=lambda x: x.get("miou_fg", 0), reverse=True)
    csv_path = os.path.join(exp_root, "results_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader(); w.writerows(all_results)
    print(f"汇总 CSV: {csv_path}")


# ==================== 主函数 ====================

def main():
    exp_root = CFG["exp_root"]
    os.makedirs(exp_root, exist_ok=True)

    print("路径检查:")
    has_error = False
    for key in ["train_img_dir", "train_mask_dir", "val_img_dir", "val_mask_dir"]:
        p = Path(CFG[key])
        if not p.exists():
            print(f"  [ERROR] {key} 不存在: {CFG[key]}")
            has_error = True
        else:
            imgs = list(p.glob("*.png"))
            print(f"  [OK]  {key}: {len(imgs)} 张  ({CFG[key]})")
            if len(imgs) == 0:
                print(f"  [ERROR] 该目录下没有 .png 文件！")
                has_error = True
    if has_error:
        print("\n请修正路径后重新运行"); sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  全监督语义分割 — 模型结构消融实验（无预训练版）")
    print(f"  共 {len(EXPERIMENTS)} 组 | CE+Dice / LR=1e-3 / 从零训练 / seed=42")
    print(f"  架构: UNet / UNet++ / DeepLabV3 / DeepLabV3+ / FPN /")
    print(f"         PSPNet / MAnet / PAN / LinkNet")
    print(f"  编码器: ResNet18/34/50 / MobileNetV2 / EfficientNet-B0")
    print(f"  每 epoch 同时输出两种 mIoU：")
    print(f"    mIoU(fg)  = 前景3类（body / solar_panel / antenna）")
    print(f"    mIoU(all) = 含背景4类")
    print(f"  早停和排名以 mIoU(fg) 为准")
    print(f"  结果目录: {exp_root}")
    print(f"{'='*65}\n")

    for pkg in ["segmentation_models_pytorch", "matplotlib"]:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  {pkg} ✓")
        except ImportError:
            if pkg == "segmentation_models_pytorch":
                print(f"  [ERROR] 请先安装: pip install {pkg}"); sys.exit(1)
            else:
                print(f"  [警告] {pkg} 未安装，曲线图和效果图将跳过")
    print()

    all_results = []
    failed_exps = []

    for i, exp_cfg in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp_cfg['name']}")

        done_flag = Path(exp_root) / exp_cfg["name"] / "done.json"
        if done_flag.exists():
            with open(done_flag) as f:
                result = json.load(f)
            print(f"  [跳过] 已完成  "
                  f"mIoU(fg)={result.get('miou_fg','?')}  "
                  f"mIoU(all)={result.get('miou_all','?')}")
            all_results.append(result)
            continue

        try:
            result = run_experiment(exp_cfg, CFG)
            all_results.append(result)
        except KeyboardInterrupt:
            print("\n  Ctrl+C 中断！checkpoint 已保存，下次运行自动续跑")
            save_summary_csv(all_results, exp_root)
            print_summary(all_results)
            sys.exit(0)
        except Exception as e:
            print(f"\n  [ERROR] {exp_cfg['name']} 失败: {e}")
            traceback.print_exc()
            failed_exps.append(exp_cfg["name"])
            err_path = Path(exp_root) / exp_cfg["name"] / "error.txt"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
            print(f"  错误日志: {err_path}  →  跳过，继续下一个...")
            continue

        save_summary_csv(all_results, exp_root)

    print_summary(all_results)
    save_summary_csv(all_results, exp_root)

    if failed_exps:
        print(f"\n[警告] 以下实验失败:")
        for name in failed_exps:
            print(f"  - {name}")

    print(f"\n✓ 全部完成！")
    print(f"  汇总 CSV  : {exp_root}/results_summary.csv")
    print(f"  各实验目录: {exp_root}/<实验名>/")
    print(f"    ├── best_model.pth   最优模型权重")
    print(f"    ├── train_log.csv    每epoch（miou_fg + miou_all 双列）")
    print(f"    ├── loss_curve.png   3图：Loss / mIoU(fg) / mIoU(all)")
    print(f"    ├── done.json        完成标志 + 双指标最终结果")
    print(f"    └── top_vis/         Top-10效果图（标题含双指标）")


if __name__ == "__main__":
    main()