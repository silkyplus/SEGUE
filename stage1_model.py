"""
stage1_model.py — SEGUE Stage 1 模型
=====================================
架构: Grounding DINO (LoRA) + Prompt Refiner + SAM2 (冻结)

Forward 流程:
1. Grounding DINO: image + text → bboxes, class scores
2. RoI Align: 在 GDINO 特征图上提取每个 bbox 的 RoI 特征
3. Prompt Refiner: RoI 特征 → 精修 bbox + 关键点
4. SAM2: image encode + prompt encode + mask decode → masks
"""

import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, box_convert

# Grounding DINO 和 SAM2 都在 D:\afss 下
AFSS_ROOT = r"D:\afss"
sys.path.insert(0, AFSS_ROOT)

# 强制 GDINO 使用 PyTorch fallback (CUDA kernel 不支持 sm_120)
import grounding_dino.groundingdino.models.GroundingDINO.ms_deform_attn as _mda
_mda.HAS_CUDA_OPS = False
print("[Compat] GDINO deformable attention → PyTorch fallback (sm_120 not supported)")

# 修复新版 transformers 中 BertModel 缺少 get_head_mask 的兼容性问题
from transformers import BertModel
if not hasattr(BertModel, 'get_head_mask'):
    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            return [head_mask] * num_hidden_layers
        return [None] * num_hidden_layers
    BertModel.get_head_mask = _get_head_mask
    print("[Compat] Patched BertModel.get_head_mask")

from grounding_dino.groundingdino.models import build_model
from grounding_dino.groundingdino.util.misc import clean_state_dict, nested_tensor_from_tensor_list
from grounding_dino.groundingdino.util.slconfig import SLConfig

from stage1_config import *


# ============================================================
# 1. Grounding DINO 加载器
# ============================================================
def load_grounding_dino(device='cuda', apply_lora_flag=True):
    """加载 Grounding DINO Swin-T，返回模型"""
    print(f"[GDINO] Loading from {GDINO_WEIGHTS}")
    args = SLConfig.fromfile(GDINO_CONFIG)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(GDINO_WEIGHTS, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)

    # 冻结 backbone + text encoder
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 可选：给 transformer decoder 加 LoRA
    if apply_lora_flag:
        model = _apply_lora_to_gdino(model)

    model = model.to(device)
    model.eval()
    print(f"[GDINO] Ready. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def _apply_lora_to_gdino(model):
    """给 Grounding DINO 的 cross-attention 层添加 LoRA"""
    try:
        from peft import LoraConfig, inject_adapter_in_model

        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias='none',
            target_modules=LORA_TARGET_MODULES,
            task_type=None,
        )
        model = inject_adapter_in_model(lora_cfg, model, adapter_name='default')
        print(f"[LoRA] Applied to GDINO decoder (r={LORA_R}, alpha={LORA_ALPHA})")
    except Exception as e:
        print(f"[LoRA] Skipped - {e}")
    return model


# ============================================================
# 2. Prompt Refiner (Image Crop based)
# ============================================================
class PromptRefiner(nn.Module):
    """
    从图像裁剪区域精修 bbox 并预测关键点。
    使用轻量 CNN 处理每个 bbox 裁剪区域，避免依赖 GDINO 内部特征。

    输入:  image_crops [N, 3, 64, 64] — 每个检测目标的图像裁剪
    输出:
      - bbox_delta [N, 4]       — 对原始 bbox 的 (dcx, dcy, dw, dh) 修正量
      - keypoints  [N, K, 2]    — K 个关键点坐标 (x, y)，归一化到 [0,1]
    """

    def __init__(self, hidden=REFINER_HIDDEN, num_keypoints=NUM_KEYPOINTS):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Lightweight CNN backbone for crop features
        self.crop_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),    # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden, 3, stride=2, padding=1), # 4x4
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                      # 1x1
        )

        # BBox 精修头: 输出 delta (dcx, dcy, dw, dh)
        self.bbox_refine_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.bbox_refine_head[-1].weight)
        nn.init.zeros_(self.bbox_refine_head[-1].bias)

        # 关键点头: 输出 K 个 (x, y) 坐标 (归一化到 [0,1])
        self.point_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_keypoints * 2),
            nn.Sigmoid(),
        )

    def forward(self, image_crops):
        """
        Args:
            image_crops: [N, 3, 64, 64]  bbox 裁剪区域

        Returns:
            bbox_delta: [N, 4]  (dcx, dcy, dw, dh) 归一化偏移量
            keypoints_norm: [N, K, 2] 归一化到 [0,1] 的关键点坐标
        """
        feat = self.crop_cnn(image_crops)              # [N, hidden, 1, 1]
        bbox_delta = self.bbox_refine_head(feat)        # [N, 4]
        points_flat = self.point_head(feat)              # [N, K*2]
        keypoints_norm = points_flat.view(-1, self.num_keypoints, 2)  # [N, K, 2]
        return bbox_delta, keypoints_norm


# ============================================================
# 3. SEGUE Stage 1 完整模型
# ============================================================
class SEGUEStage1(nn.Module):
    """
    SEGUE Stage 1: Grounding Pretrain

    Forward:
        image + text → Grounding DINO → bboxes
        → RoI Align → Prompt Refiner → refined bbox + keypoints
        → SAM2 → masks

    Loss (外部计算):
        L_geom: L1(bbox) + GIoU(bbox) + L2(keypoints vs GT)
        L_align: Dice(mask) + BCE(mask) vs GT mask
    """

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # ---- Grounding DINO ----
        print("\n" + "=" * 60)
        print("  Loading Grounding DINO...")
        print("=" * 60)
        self.gdino = load_grounding_dino(device=device, apply_lora_flag=USE_LORA)

        # ---- Prompt Refiner ----
        self.prompt_refiner = PromptRefiner(
            hidden=REFINER_HIDDEN,
            num_keypoints=NUM_KEYPOINTS,
        ).to(device)
        print(f"[Refiner] Trainable params: {sum(p.numel() for p in self.prompt_refiner.parameters()):,}")

        # ---- SAM2 ----
        print("\n" + "=" * 60)
        print("  Loading SAM2...")
        print("=" * 60)
        self.sam2_model, self.sam2_predictor = self._load_sam2(device)
        # 冻结 SAM2 全部参数
        for p in self.sam2_model.parameters():
            p.requires_grad = False
        print(f"[SAM2] Frozen. Trainable params: {sum(p.numel() for p in self.sam2_model.parameters() if p.requires_grad):,}")

        self.sam2_loaded = True

    def _load_sam2(self, device):
        """加载 SAM2 模型"""
        import os
        sam2_dir = os.path.join(SAM2_ROOT, "sam2")
        orig_cwd = os.getcwd()
        os.chdir(sam2_dir)

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(
            config_file=SAM2_CONFIG,
            ckpt_path=SAM2_CHECKPOINT,
            device=device,
        )
        predictor = SAM2ImagePredictor(model)

        os.chdir(orig_cwd)
        return model, predictor

    def forward(self, images, captions, gt_bboxes=None, gt_masks=None, image_sizes=None,
                run_sam2=False):
        """
        Args:
            images:      [B, 3, H, W]
            captions:    List[str]
            run_sam2:    bool — 训练时设 False (只用 L_geom)，验证时设 True
        """
        B, C, H, W = images.shape

        # Step 1: GDINO → bboxes
        gdino_boxes_list, gdino_scores_list, gdino_phrases_list = self._forward_gdino(
            images, captions
        )

        # 反归一化图像用于裁剪
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        images_unnorm = images * std + mean
        images_unnorm = torch.clamp(images_unnorm, 0, 1)

        pred_bboxes_list, pred_keypoints_list, pred_masks_list = [], [], []

        for b in range(B):
            img_b_unnorm = images_unnorm[b]
            img_H, img_W = H, W
            gdino_boxes = gdino_boxes_list[b]

            if len(gdino_boxes) == 0:
                pred_bboxes_list.append(torch.zeros(0, 4, device=self.device))
                pred_keypoints_list.append(torch.zeros(0, NUM_KEYPOINTS, 2, device=self.device))
                pred_masks_list.append(torch.zeros(0, H, W, device=self.device))
                continue

            # Step 2: Crop → Prompt Refiner
            image_crops = self._crop_regions(img_b_unnorm, gdino_boxes, crop_size=64)
            bbox_delta, keypoints_norm = self.prompt_refiner(image_crops)

            refined_boxes = gdino_boxes + 0.1 * torch.tanh(bbox_delta)
            refined_boxes = torch.clamp(refined_boxes, 0.0, 1.0)

            keypoints_pixel = keypoints_norm.clone()
            keypoints_pixel[..., 0] *= img_W
            keypoints_pixel[..., 1] *= img_H

            # Step 3: SAM2 (only if requested, e.g. validation)
            if run_sam2:
                masks_b = self._forward_sam2(
                    images[b], refined_boxes, keypoints_pixel,
                    original_size=(img_H, img_W),
                )
            else:
                masks_b = torch.zeros(len(gdino_boxes), H, W, device=self.device)

            pred_bboxes_list.append(refined_boxes)
            pred_keypoints_list.append(keypoints_pixel)
            pred_masks_list.append(masks_b)

        return {
            'pred_bboxes': pred_bboxes_list,
            'pred_keypoints': pred_keypoints_list,
            'pred_masks': pred_masks_list,
            'gdino_bboxes': gdino_boxes_list,
            'gdino_scores': gdino_scores_list,
            'gdino_phrases': gdino_phrases_list,
        }

    def _crop_regions(self, image, boxes_cxcywh, crop_size=64):
        """从图像中裁剪 bbox 区域，返回统一尺寸的 crops"""
        N = boxes_cxcywh.shape[0]
        _, H, W = image.shape
        crops = []

        xyxy = box_convert(boxes_cxcywh, in_fmt='cxcywh', out_fmt='xyxy')
        for i in range(N):
            x1, y1, x2, y2 = xyxy[i]
            # 转为像素坐标
            x1_px = int(x1 * W)
            y1_px = int(y1 * H)
            x2_px = int(x2 * W)
            y2_px = int(y2 * H)

            # 扩展边界
            pad_w = max(0, int((x2_px - x1_px) * 0.2))
            pad_h = max(0, int((y2_px - y1_px) * 0.2))
            x1_px = max(0, x1_px - pad_w)
            y1_px = max(0, y1_px - pad_h)
            x2_px = min(W, x2_px + pad_w)
            y2_px = min(H, y2_px + pad_h)

            crop = image[:, y1_px:y2_px, x1_px:x2_px]
            crop = F.interpolate(crop.unsqueeze(0), size=(crop_size, crop_size),
                                mode='bilinear', align_corners=False).squeeze(0)
            crops.append(crop)

        if crops:
            return torch.stack(crops, dim=0)  # [N, 3, 64, 64]
        return torch.zeros(0, 3, crop_size, crop_size, device=image.device)

    def _forward_gdino(self, images, captions):
        """
        Grounding DINO forward (简化版: 使用标准 forward API)。

        Returns:
            boxes_list:  List[Tensor [N_i, 4]] cxcywh 归一化
            scores_list: List[Tensor [N_i]]
            phrases_list:List[List[str]]
        """
        boxes_list, scores_list, phrases_list = [], [], []

        for b in range(images.shape[0]):
            img_b = images[b:b+1]
            cap_b = captions[b] if isinstance(captions, list) else captions

            with torch.set_grad_enabled(USE_LORA):
                samples = nested_tensor_from_tensor_list(img_b)
                out = self.gdino(samples, captions=[cap_b])

            pred_logits = out['pred_logits'].sigmoid()[0]  # [900, 256]
            pred_boxes = out['pred_boxes'][0]              # [900, 4]

            scores = pred_logits.max(dim=1)[0]
            mask = scores > GDINO_BOX_THRESHOLD

            filtered_boxes = pred_boxes[mask]
            filtered_scores = scores[mask]
            filtered_logits = pred_logits[mask]

            # Extract phrases
            from grounding_dino.groundingdino.util.utils import get_phrases_from_posmap
            tokenized = self.gdino.tokenizer(cap_b)
            phrases = [
                get_phrases_from_posmap(
                    logit > GDINO_TEXT_THRESHOLD,
                    tokenized, self.gdino.tokenizer
                ).replace('.', '')
                for logit in filtered_logits
            ]

            boxes_list.append(filtered_boxes)
            scores_list.append(filtered_scores)
            phrases_list.append(phrases)

        return boxes_list, scores_list, phrases_list

    def _forward_sam2(self, image, bboxes_cxcywh, keypoints_pixel, original_size):
        """
        使用 SAM2 生成 masks。

        Args:
            image:          [3, H, W] 单张图 (已归一化)
            bboxes_cxcywh:  [N, 4] cxcywh 归一化坐标
            keypoints_pixel:[N, K, 2] 关键点像素坐标
            original_size:  (H, W) 原图尺寸

        Returns:
            masks: [N, H, W] mask logits
        """
        import os
        orig_cwd = os.getcwd()
        sam2_dir = os.path.join(SAM2_ROOT, "sam2")
        os.chdir(sam2_dir)

        H, W = original_size
        N = bboxes_cxcywh.shape[0]

        # 预处理图像到 SAM2 格式
        # SAM2 predictor.set_image 会处理 resize
        # 我们直接操作 SAM2 model
        from sam2.utils.transforms import SAM2Transforms

        # Resize 图像到 SAM2 输入尺寸，保留宽高比
        transform = SAM2Transforms(
            resolution=SAM2_IMAGE_SIZE,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )

        # SAM2 期望 [H, W, 3] numpy uint8
        # 反归一化
        mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
        img_unnorm = image * std + mean
        img_unnorm = torch.clamp(img_unnorm, 0, 1)
        img_np = (img_unnorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # SAM2 predict
        self.sam2_predictor.set_image(img_np)

        masks_list = []
        for i in range(N):
            bbox_cxcywh = bboxes_cxcywh[i].detach().cpu().numpy()
            # cxcywh 归一化 → xyxy 像素坐标
            cx, cy, w, h = bbox_cxcywh
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H
            bbox_xyxy = np.array([x1, y1, x2, y2])

            # 关键点
            pts = keypoints_pixel[i].detach().cpu().numpy()
            pts_labels = np.ones(len(pts), dtype=np.int32)

            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=pts,
                point_labels=pts_labels,
                box=bbox_xyxy,
                multimask_output=False,
            )
            mask_tensor = torch.from_numpy(masks[0].astype(np.float32)).to(self.device)
            masks_list.append(mask_tensor)

        os.chdir(orig_cwd)

        if masks_list:
            return torch.stack(masks_list, dim=0)  # [N, H, W]
        else:
            return torch.zeros(0, H, W, device=self.device)

    def train(self, mode=True):
        super().train(mode)
        # SAM2 始终 eval
        self.sam2_model.eval()
        return self


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = SEGUEStage1(device=device)
    model.eval()

    # 测试 forward
    x = torch.randn(1, 3, 512, 512, device=device)
    captions = ["antenna. body. solar_panel."]

    print("\nTesting forward...")
    with torch.no_grad():
        out = model(x, captions)

    for k, v in out.items():
        if isinstance(v, list):
            print(f"  {k}: len={len(v)}, shape[0]={v[0].shape if len(v) > 0 and hasattr(v[0], 'shape') else 'N/A'}")
        else:
            print(f"  {k}: shape={v.shape if hasattr(v, 'shape') else v}")

    print("\n✓ Model test passed!")
