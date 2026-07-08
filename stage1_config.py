"""
stage1_config.py — SEGUE Stage 1 配置文件
===========================================
Stage 1: Grounding Pretrain
- 冻结 Grounding DINO backbone + text encoder
- LoRA 微调 transformer decoder (可选)
- 训练 Prompt Refiner (bbox 精修 + 关键点预测)
- 冻结 SAM2
- Loss: L_geom + L_align
"""

from pathlib import Path

# ==============================================================
# Grounding DINO 路径
# ==============================================================
GDINO_ROOT = r"D:\afss"
GDINO_REPO = r"D:\afss\grounding_dino"
GDINO_CONFIG = r"D:\afss\grounding_dino\groundingdino\config\GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = r"D:\afss\gdino_checkpoints\groundingdino_swint_ogc.pth"

# ==============================================================
# SAM2 路径
# ==============================================================
SAM2_ROOT = r"D:\afss"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CHECKPOINT = r"D:\pycharm_projects\NOTRAING\NOTRAING\checkpoints\sam2.1_hiera_base_plus.pt"

# ==============================================================
# 数据集 (先用卫星数据集快速验证)
# ==============================================================
DATA_ROOT = r"D:\data\final_dataset\final_dataset"
TRAIN_IMG_DIR = Path(DATA_ROOT) / "images" / "train"
TRAIN_MASK_DIR = Path(DATA_ROOT) / "masks" / "train"
VAL_IMG_DIR = Path(DATA_ROOT) / "images" / "val"
VAL_MASK_DIR = Path(DATA_ROOT) / "masks" / "val"

# ==============================================================
# 类别定义
# ==============================================================
CLASS_NAMES = ['body', 'solar_panel', 'antenna']
NUM_CLASSES = len(CLASS_NAMES)

# GT mask RGB -> 类别索引
RGB_TO_CLASS = {
    (0, 0, 0): 0,        # background
    (0, 255, 0): 1,      # body
    (255, 0, 0): 2,      # solar_panel
    (0, 0, 255): 3,      # antenna
}

# 文本查询格式: 类别名用 ". " 连接 (Grounding DINO 格式)
TEXT_QUERY = ". ".join(CLASS_NAMES) + "."

# ==============================================================
# 模型超参
# ==============================================================
# Grounding DINO
GDINO_HIDDEN_DIM = 256
GDINO_BOX_THRESHOLD = 0.25   # 训练时放宽阈值，保留更多候选
GDINO_TEXT_THRESHOLD = 0.2
GDINO_MAX_OBJECTS = 20       # 每张图最多检测目标数

# Prompt Refiner
REFINER_HIDDEN = 256
NUM_KEYPOINTS = 2            # 每个目标预测的关键点数量

# SAM2
SAM2_IMAGE_SIZE = 1024       # SAM2 内部处理尺寸

# LoRA 配置 (给 Grounding DINO decoder)
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["qkv", "proj"]  # Swin-T 的 attention 模块

# ==============================================================
# 损失权重
# ==============================================================
L1_WEIGHT = 1.0              # bbox L1 loss
GIOU_WEIGHT = 2.0            # bbox GIoU loss
POINT_WEIGHT = 0.5           # 关键点 L2 loss
DICE_WEIGHT = 1.0            # mask Dice loss
BCE_WEIGHT = 1.0             # mask BCE loss

# ==============================================================
# 训练超参
# ==============================================================
BATCH_SIZE = 2               # SAM2 比较吃显存
NUM_WORKERS = 2
EPOCHS = 50
LR = 1e-4                    # Prompt Refiner + LoRA 学习率
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
GRAD_CLIP = 1.0
PATIENCE = 10                # Early stopping

# 图像尺寸
TRAIN_CROP = 512             # 训练时裁剪尺寸 (必须 >= 512 且是 16 的倍数)
VAL_SIZE = 512

# ImageNet 归一化
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ==============================================================
# 输出
# ==============================================================
OUTPUT_DIR = Path(r"D:\pycharm_projects\NOTRAING\NOTRAING\SEGUE\runs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================
# 日志
# ==============================================================
LOG_EVERY = 10
VAL_EVERY = 2
SEED = 42
