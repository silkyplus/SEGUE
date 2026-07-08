"""
stage1_dataset.py — SEGUE Stage 1 数据集
=========================================
支持:
1. 卫星数据集 (RGB mask → bbox/points/masks)
2. COCO (category → bbox/points/masks)
3. RefCOCO (referring expression → bbox/mask)

当前实现: 卫星数据集适配器
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random

from stage1_config import *
from stage1_losses import masks_to_bboxes_and_points


# ============================================================
# 卫星数据集 - 改造为 Grounding 格式
# ============================================================
class SatelliteGroundingDataset(Dataset):
    """
    卫星分割数据集 → Grounding 格式

    每张图可能有多个类别的目标，返回:
    - image: [3, H, W] 归一化张量
    - caption: str  文本查询
    - gt_masks: [N, H, W] binary masks
    - gt_bboxes: [N, 4] cxcywh 归一化
    - gt_points: [N, 2, 2] 关键点像素坐标
    """

    def __init__(self, img_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR,
                 crop=TRAIN_CROP, mask_suffix='_mask.png', is_train=True):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.crop = crop
        self.is_train = is_train

        imgs = sorted(list(self.img_dir.glob('*.png')) + list(self.img_dir.glob('*.jpg')))
        self.items = []
        for p in imgs:
            mp = self.mask_dir / f"{p.stem}{mask_suffix}"
            if mp.exists():
                self.items.append((p, mp))

        if len(self.items) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")
        print(f"[Dataset] {len(self.items)} pairs from {img_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]

        # 加载图像和 mask
        img = Image.open(img_path).convert('RGB')
        msk = Image.open(mask_path).convert('RGB')
        img_np = np.asarray(img)
        msk_np = np.asarray(msk)

        # RGB mask → class index map
        cls_map = self._rgb_to_class(msk_np)  # [H, W] int

        # 训练增强
        if self.is_train:
            img_np, cls_map = self._augment(img_np, cls_map)

        H, W = img_np.shape[:2]

        # 确保尺寸是 16 的倍数 (SAM2 要求)
        H = (H // 16) * 16
        W = (W // 16) * 16
        img_np = img_np[:H, :W]
        cls_map = cls_map[:H, :W]

        # 提取每个类别的 binary mask
        gt_masks = []
        gt_labels = []

        for cls_idx in range(1, NUM_CLASSES + 1):  # 跳过 background (0)
            binary_mask = (cls_map == cls_idx)
            if binary_mask.sum() > 16:  # 至少 16 个像素
                gt_masks.append(torch.from_numpy(binary_mask).float())
                gt_labels.append(cls_idx - 1)  # 0-indexed

        # 生成 caption
        # 使用所有可能类别，或者只使用图中存在的类别
        present_classes = [CLASS_NAMES[l] for l in gt_labels]
        if present_classes:
            caption = ". ".join(present_classes) + "."
        else:
            caption = TEXT_QUERY  # fallback: 所有类别

        # 转换为张量
        img_t = torch.from_numpy(img_np.copy()).float().permute(2, 0, 1) / 255.0
        img_t = TF.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        if len(gt_masks) > 0:
            gt_masks_t = torch.stack(gt_masks, dim=0)  # [N, H, W]
            gt_bboxes_t, gt_points_t = masks_to_bboxes_and_points(gt_masks_t)
        else:
            gt_masks_t = torch.zeros(0, H, W)
            gt_bboxes_t = torch.zeros(0, 4)
            gt_points_t = torch.zeros(0, NUM_KEYPOINTS, 2)

        return {
            'image': img_t,
            'caption': caption,
            'gt_masks': gt_masks_t,
            'gt_bboxes': gt_bboxes_t,
            'gt_points': gt_points_t,
            'gt_labels': torch.tensor(gt_labels) if gt_labels else torch.tensor([]),
            'image_size': (H, W),
            'stem': img_path.stem,
        }

    def _rgb_to_class(self, mask_rgb):
        """RGB mask → class index map"""
        H, W, _ = mask_rgb.shape
        out = np.zeros((H, W), dtype=np.int64)
        for rgb, cls in RGB_TO_CLASS.items():
            if cls == 0:
                continue
            r, g, b = rgb
            match = (mask_rgb[..., 0] == r) & (mask_rgb[..., 1] == g) & (mask_rgb[..., 2] == b)
            out[match] = cls
        return out

    def _augment(self, img, cls_map):
        """训练增强"""
        H, W = img.shape[:2]
        crop = self.crop

        # 随机裁剪 (如果图比 crop 大)
        if H > crop and W > crop:
            y0 = random.randint(0, H - crop)
            x0 = random.randint(0, W - crop)
            img = img[y0:y0 + crop, x0:x0 + crop]
            cls_map = cls_map[y0:y0 + crop, x0:x0 + crop]
        elif H != crop or W != crop:
            # resize
            img = TF.resize(
                torch.from_numpy(img).permute(2, 0, 1),
                (crop, crop),
                interpolation=TF.InterpolationMode.BILINEAR
            ).permute(1, 2, 0).numpy()
            # 用最近邻 resize cls_map
            from PIL import Image as PILImage
            cls_map_resized = PILImage.fromarray(cls_map.astype(np.uint8)).resize(
                (crop, crop), resample=PILImage.NEAREST
            )
            cls_map = np.array(cls_map_resized)

        # 随机水平翻转
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            cls_map = np.ascontiguousarray(cls_map[:, ::-1])

        # 随机 90° 旋转
        if random.random() < 0.5:
            k = random.randint(0, 3)
            img = np.ascontiguousarray(np.rot90(img, k))
            cls_map = np.ascontiguousarray(np.rot90(cls_map, k))

        # 轻颜色抖动
        if random.random() < 0.5:
            b = 1.0 + random.uniform(-0.1, 0.1)
            img = np.clip(img.astype(np.float32) * b, 0, 255).astype(np.uint8)

        return img, cls_map


# ============================================================
# 批处理 collate
# ============================================================
def grounding_collate_fn(batch):
    """
    自定义 collate，处理变长 list (不同数量的目标)
    """
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    gt_masks = [item['gt_masks'] for item in batch]
    gt_bboxes = [item['gt_bboxes'] for item in batch]
    gt_points = [item['gt_points'] for item in batch]
    gt_labels = [item['gt_labels'] for item in batch]
    image_sizes = [item['image_size'] for item in batch]
    stems = [item['stem'] for item in batch]

    return {
        'image': images,
        'caption': captions,
        'gt_masks': gt_masks,
        'gt_bboxes': gt_bboxes,
        'gt_points': gt_points,
        'gt_labels': gt_labels,
        'image_size': image_sizes,
        'stem': stems,
    }


# ============================================================
# 验证数据集 (不做增强)
# ============================================================
class SatelliteGroundingValDataset(SatelliteGroundingDataset):
    def __init__(self, img_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR,
                 crop=VAL_SIZE, mask_suffix='_mask.png'):
        super().__init__(img_dir=img_dir, mask_dir=mask_dir,
                         crop=crop, mask_suffix=mask_suffix, is_train=False)


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Dataset 自测")
    print("=" * 50)

    ds = SatelliteGroundingDataset(is_train=True)
    print(f"\nDataset size: {len(ds)}")

    if len(ds) > 0:
        item = ds[0]
        print(f"\nSample item:")
        print(f"  image:     {item['image'].shape}  {item['image'].dtype}")
        print(f"  caption:   {item['caption']}")
        print(f"  gt_masks:  {item['gt_masks'].shape}")
        print(f"  gt_bboxes: {item['gt_bboxes'].shape}")
        print(f"  gt_points: {item['gt_points'].shape}")
        print(f"  gt_labels: {item['gt_labels']}")
        print(f"  size:      {item['image_size']}")

        # 检查 bbox 的有效性
        if len(item['gt_bboxes']) > 0:
            print(f"\n  BBoxes (cxcywh norm):")
            for i, b in enumerate(item['gt_bboxes']):
                name = CLASS_NAMES[int(item['gt_labels'][i])]
                print(f"    {name}: cx={b[0]:.3f}, cy={b[1]:.3f}, w={b[2]:.3f}, h={b[3]:.3f}")

    print("\n✓ Dataset test passed!")
