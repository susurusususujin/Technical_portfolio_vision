# train_quality_cv.py
# ============================================================
# Intelligent Pest Image Quality Classification (OK vs ERROR)
#  - Stratified 5-Fold CV
#  - Transfer Learning (EfficientNet-B0 / MobileNetV3 / ResNet18)
#  - Early Stopping
#  - Threshold auto-selection on VAL to satisfy target ERROR recall
#  - Reports Precision/Recall/F1 for ERROR class (re-shoot decision)
#
# ✅ YOUR folder structure:
#   NARAE_TREND/
#     image/
#       true_image/   <- 정상 이미지 (OK)
#       false_image/  <- 오류 이미지 (ERROR)
#     train_quality_cv.py
#     outputs/
#
# Output:
#  - per-fold best model saved
#  - per-fold metrics and selected threshold printed
# ============================================================

import os
import glob
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # (옵션) 그래도 보통은 try/except가 핵심


try:
    from sklearn.model_selection import StratifiedKFold
except ImportError as e:
    raise ImportError("scikit-learn이 필요합니다. pip install scikit-learn") from e

try:
    import torchvision
    import torchvision.transforms as T
    from torchvision.models import (
        efficientnet_b0, EfficientNet_B0_Weights,
        mobilenet_v3_large, MobileNet_V3_Large_Weights,
        resnet18, ResNet18_Weights,
    )
except ImportError as e:
    raise ImportError("torchvision이 필요합니다. pip install torchvision") from e


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Dataset
# -------------------------
class QualityDataset(Dataset):
    """
    Labels:
      OK (true_image folder)    -> 0
      ERROR (false_image folder)-> 1  (positive class)
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # ✅ CHANGED: folder names
        true_dir = os.path.join(root_dir, "true_image")
        false_dir = os.path.join(root_dir, "false_image")

        if not os.path.isdir(true_dir) or not os.path.isdir(false_dir):
            raise FileNotFoundError(
                f"폴더 구조가 필요합니다:\n"
                f"{root_dir}/true_image, {root_dir}/false_image\n"
                f"현재: true_image exists={os.path.isdir(true_dir)}, false_image exists={os.path.isdir(false_dir)}"
            )

        true_files = sorted(self._list_images(true_dir))
        false_files = sorted(self._list_images(false_dir))

        # (path, label)
        self.samples: List[Tuple[str, int]] = [(p, 0) for p in true_files] + [(p, 1) for p in false_files]

        if len(self.samples) == 0:
            raise ValueError("이미지 파일을 찾지 못했습니다. (jpg/png/jpeg/bmp/tif/tiff/webp)")

    @staticmethod
    def _list_images(d: str) -> List[str]:
        # ✅ include uppercase too (windows 흔함)
        exts = (
            "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp",
            "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP",
        )
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(d, e)))
        return files

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # 깨진/손상 이미지면 "대체 샘플"로 넘어감 (재귀로 다음 샘플)
            # 무한루프 방지: 최대 10번까지만 점프
            for _ in range(10):
                idx = random.randint(0, len(self.samples) - 1)
                path, label = self.samples[idx]
                try:
                    img = Image.open(path).convert("RGB")
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(f"Too many corrupted images encountered. Last path={path}") from e

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32), path



# -------------------------
# Model builder
# -------------------------
def build_model(model_name: str, num_classes: int = 1):
    """
    Returns model outputting logits of shape [B, 1].
    """
    name = model_name.lower().strip()

    if name in ["effb0", "efficientnet_b0", "efficientnet"]:
        weights = EfficientNet_B0_Weights.DEFAULT
        m = efficientnet_b0(weights=weights)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        return m, weights.transforms()

    if name in ["mobilenetv3", "mobilenet_v3_large", "mbv3"]:
        weights = MobileNet_V3_Large_Weights.DEFAULT
        m = mobilenet_v3_large(weights=weights)
        in_features = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_features, num_classes)
        return m, weights.transforms()

    if name in ["resnet18", "resnet"]:
        weights = ResNet18_Weights.DEFAULT
        m = resnet18(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m, weights.transforms()

    raise ValueError("model_name은 effb0 / mobilenetv3 / resnet18 중 하나를 추천합니다.")


# -------------------------
# Metrics & threshold search
# -------------------------
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

def compute_prf_for_positive(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    """
    Positive class = ERROR (label=1)
    """
    y_pred = (y_prob >= thr).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

def find_threshold_auto(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.95,
    grid_step: float = 0.01
) -> Tuple[float, Dict[str, float]]:
    """
    Choose threshold on validation set:
      1) among thresholds with recall >= target_recall, pick the one with best precision
         (tie-break: higher F1)
      2) if none meet target recall, pick threshold with best recall, then best precision
    """
    best_thr = 0.5
    best = None
    candidates = []

    for thr in np.arange(0.0, 1.0 + 1e-9, grid_step):
        m = compute_prf_for_positive(y_true, y_prob, float(thr))
        candidates.append((thr, m))

    feasible = [(thr, m) for thr, m in candidates if m["recall"] >= target_recall]
    if feasible:
        feasible.sort(key=lambda x: (x[1]["precision"], x[1]["f1"]), reverse=True)
        best_thr, best = feasible[0]
        return float(best_thr), best

    candidates.sort(key=lambda x: (x[1]["recall"], x[1]["precision"], x[1]["f1"]), reverse=True)
    best_thr, best = candidates[0]
    return float(best_thr), best


# -------------------------
# Train/Eval
# -------------------------
@dataclass
class TrainConfig:
    data_dir: str
    model_name: str
    image_size: int = 224
    batch_size: int = 16
    epochs_stage1: int = 8
    epochs_stage2: int = 12
    lr_stage1: float = 3e-4
    lr_stage2: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    k_folds: int = 5
    seed: int = 42
    patience: int = 5
    target_recall_error: float = 0.95
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "outputs"


def make_transforms(image_size: int):
    train_tf = T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomResizedCrop(image_size, scale=(0.80, 1.00)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    val_tf = T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def freeze_backbone(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any(k in name for k in ["classifier", "fc"]):
            p.requires_grad = True


def unfreeze_top_layers(model: nn.Module, model_name: str):
    for p in model.parameters():
        p.requires_grad = True


def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def get_pos_weight(labels: np.ndarray) -> torch.Tensor:
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(float(neg) / float(pos))


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    all_probs = []
    all_true = []
    all_paths = []
    for x, y, paths in loader:
        x = x.to(device)
        logits = model(x).squeeze(1)  # [B]
        probs = sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_true.append(y.numpy())
        all_paths.extend(list(paths))
    return np.concatenate(all_true), np.concatenate(all_probs), all_paths


def train_one_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    optimizer,
    criterion,
    max_epochs: int,
    patience: int
):
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * x.size(0)
            n += x.size(0)

        train_loss = running / max(n, 1)

        model.eval()
        v_running = 0.0
        v_n = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)
                v_running += float(loss.item()) * x.size(0)
                v_n += x.size(0)
        val_loss = v_running / max(v_n, 1)

        print(f"    epoch {epoch:02d}/{max_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    early stopping (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    # ✅ CHANGED: default data_dir to "image" (your structure)
    ap.add_argument("--data_dir", type=str, default="image", help='dataset root (contains "true_image"/"false_image")')
    ap.add_argument("--model", type=str, default="effb0", help="effb0 | mobilenetv3 | resnet18")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_recall_error", type=float, default=0.95)
    ap.add_argument("--save_dir", type=str, default="outputs")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        model_name=args.model,
        image_size=args.img_size,
        batch_size=args.batch_size,
        k_folds=args.k_folds,
        seed=args.seed,
        target_recall_error=args.target_recall_error,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
    )

    os.makedirs(cfg.save_dir, exist_ok=True)
    seed_everything(cfg.seed)
    device = cfg.device
    print(f"[INFO] device = {device}")

    train_tf, val_tf = make_transforms(cfg.image_size)

    base_ds = QualityDataset(cfg.data_dir, transform=None)
    labels = np.array([lab for _, lab in base_ds.samples], dtype=np.int32)
    idxs = np.arange(len(base_ds))

    print(f"[INFO] total={len(base_ds)} | OK(true_image)= {(labels==0).sum()} | ERROR(false_image)= {(labels==1).sum()}")
    print(f"[INFO] positive class = ERROR(false_image) = 1 (re-shoot target)")

    skf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(idxs, labels), start=1):
        print("\n" + "=" * 70)
        print(f"[FOLD {fold}/{cfg.k_folds}]")

        train_ds = QualityDataset(cfg.data_dir, transform=train_tf)
        valtest_ds = QualityDataset(cfg.data_dir, transform=val_tf)

        train_idx_list = train_idx.tolist()
        rng = np.random.RandomState(cfg.seed + fold)

        cls0 = [i for i in train_idx_list if labels[i] == 0]
        cls1 = [i for i in train_idx_list if labels[i] == 1]
        rng.shuffle(cls0)
        rng.shuffle(cls1)
        val0_n = max(1, int(0.15 * len(cls0)))
        val1_n = max(1, int(0.15 * len(cls1)))
        val_idx = np.array(cls0[:val0_n] + cls1[:val1_n], dtype=np.int64)
        tr_idx = np.array(cls0[val0_n:] + cls1[val1_n:], dtype=np.int64)

        print(f"  train={len(tr_idx)} | val={len(val_idx)} | test={len(test_idx)}")

        train_loader = DataLoader(
            Subset(train_ds, tr_idx),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device.startswith("cuda")),
        )
        val_loader = DataLoader(
            Subset(valtest_ds, val_idx),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.startswith("cuda")),
        )
        test_loader = DataLoader(
            Subset(valtest_ds, test_idx),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.startswith("cuda")),
        )

        model, _ = build_model(cfg.model_name, num_classes=1)
        model = model.to(device)

        pos_weight = get_pos_weight(labels[tr_idx]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        print("  [Stage1] freeze backbone, train head")
        freeze_backbone(model)
        opt1 = make_optimizer(model, lr=cfg.lr_stage1, weight_decay=cfg.weight_decay)
        model = train_one_stage(
            model, train_loader, val_loader, device, opt1, criterion,
            max_epochs=cfg.epochs_stage1, patience=cfg.patience
        )

        print("  [Stage2] fine-tune (unfreeze), small LR")
        unfreeze_top_layers(model, cfg.model_name)
        opt2 = make_optimizer(model, lr=cfg.lr_stage2, weight_decay=cfg.weight_decay)
        model = train_one_stage(
            model, train_loader, val_loader, device, opt2, criterion,
            max_epochs=cfg.epochs_stage2, patience=cfg.patience
        )

        yv_true, yv_prob, _ = predict_probs(model, val_loader, device)
        thr, thr_metrics = find_threshold_auto(
            y_true=yv_true.astype(np.int32),
            y_prob=yv_prob.astype(np.float32),
            target_recall=cfg.target_recall_error,
            grid_step=0.01
        )
        print(f"  [VAL] selected_thr={thr:.2f} | P={thr_metrics['precision']:.3f} R={thr_metrics['recall']:.3f} F1={thr_metrics['f1']:.3f}")

        yt_true, yt_prob, _ = predict_probs(model, test_loader, device)
        test_m = compute_prf_for_positive(yt_true.astype(np.int32), yt_prob.astype(np.float32), thr)

        print(f"  [TEST] P={test_m['precision']:.3f} R={test_m['recall']:.3f} F1={test_m['f1']:.3f} | TP={test_m['tp']} FP={test_m['fp']} FN={test_m['fn']}")

        save_path = os.path.join(cfg.save_dir, f"fold{fold}_{cfg.model_name}_thr{thr:.2f}.pt")
        torch.save({
            "model_name": cfg.model_name,
            "state_dict": model.state_dict(),
            "threshold": thr,
            "cfg": cfg.__dict__,
        }, save_path)
        print(f"  [SAVE] {save_path}")

        fold_results.append({
            "fold": fold,
            "thr": thr,
            "precision": test_m["precision"],
            "recall": test_m["recall"],
            "f1": test_m["f1"],
            "tp": test_m["tp"],
            "fp": test_m["fp"],
            "fn": test_m["fn"],
        })

    p = np.array([r["precision"] for r in fold_results], dtype=np.float64)
    r_ = np.array([r["recall"] for r in fold_results], dtype=np.float64)
    f = np.array([r["f1"] for r in fold_results], dtype=np.float64)
    th = np.array([r["thr"] for r in fold_results], dtype=np.float64)

    print("\n" + "=" * 70)
    print("[SUMMARY] (Positive=ERROR) 5-fold mean ± std")
    print(f"  threshold: {th.mean():.2f} ± {th.std(ddof=1):.2f}")
    print(f"  precision: {p.mean():.3f} ± {p.std(ddof=1):.3f}")
    print(f"  recall   : {r_.mean():.3f} ± {r_.std(ddof=1):.3f}")
    print(f"  f1       : {f.mean():.3f} ± {f.std(ddof=1):.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
