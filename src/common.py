from __future__ import annotations

import io
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Воспроизводимость
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Единая фиксация всех источников случайности (требование чекпойнта)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --------------------------------------------------------------------------- #
# Данные
# --------------------------------------------------------------------------- #
def _remap_path(csv_path: str, images_dir: Path) -> Path:
    """
    CSV хранят абсолютные пути с чужой машины (Google Drive).
    Берём только имя файла и привязываем к локальной images_dir, чтобы
    пайплайн переносился между средами без правки CSV.
    """
    return images_dir / Path(csv_path).name


class SkinLesionDataset(Dataset):
    """Датасет кожных новообразований ISIC-2018 / HAM10000."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        class_to_idx: Dict[str, int],
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = _remap_path(row["path"], self.images_dir)
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        return image, label


def build_transforms(image_size: int, augmentation: str = "light"):
    """light — только resize/normalize; strong — расширенные аугментации."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    if augmentation == "strong":
        train_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:  # light
        train_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return train_tf, eval_tf


def load_splits(splits_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits_dir = Path(splits_dir)
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    return train_df, val_df, test_df


def make_loaders(cfg, classes: Sequence[str]):
    """Собирает train/val/test даталоадеры из готовых CSV-сплитов."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    images_dir = Path(cfg.data.images_dir)
    train_df, val_df, test_df = load_splits(cfg.data.splits_dir)

    train_tf, eval_tf = build_transforms(cfg.data.image_size, cfg.train.augmentation)

    train_ds = SkinLesionDataset(train_df, images_dir, class_to_idx, train_tf)
    val_ds = SkinLesionDataset(val_df, images_dir, class_to_idx, eval_tf)
    test_ds = SkinLesionDataset(test_df, images_dir, class_to_idx, eval_tf)

    nw = cfg.data.num_workers
    bs = cfg.train.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)
    return train_loader, val_loader, test_loader, (train_df, val_df, test_df)


def make_loaders_explicit(splits_dir, images_dir, classes, *, image_size=224,
                          augmentation="light", batch_size=32, num_workers=2):
    """Лоадеры с явно заданными аугментацией и batch_size (для перебора экспериментов)."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    images_dir = Path(images_dir)
    train_df, val_df, test_df = load_splits(splits_dir)
    train_tf, eval_tf = build_transforms(image_size, augmentation)
    train_ds = SkinLesionDataset(train_df, images_dir, class_to_idx, train_tf)
    val_ds = SkinLesionDataset(val_df, images_dir, class_to_idx, eval_tf)
    test_ds = SkinLesionDataset(test_df, images_dir, class_to_idx, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, (train_df, val_df, test_df)


def compute_class_weights(train_df: pd.DataFrame, classes: Sequence[str]) -> torch.Tensor:
    from sklearn.utils.class_weight import compute_class_weight

    y = train_df["label"].map({c: i for i, c in enumerate(classes)}).values
    weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(classes)), y=y
    )
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Модели
# --------------------------------------------------------------------------- #
def _freeze(model: nn.Module, trainable: bool = False) -> None:
    for p in model.parameters():
        p.requires_grad = trainable


def build_vit_b16(num_classes: int, pretrained=True, unfreeze_last_blocks=2, dropout=0.3):
    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = models.vit_b_16(weights=weights)
    _freeze(model, trainable=False)
    if unfreeze_last_blocks > 0:
        for layer in model.encoder.layers[-unfreeze_last_blocks:]:
            _freeze(layer, trainable=True)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def build_resnet18(num_classes: int, pretrained=True, dropout=0.2):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    _freeze(model, trainable=False)  
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def build_resnet18_finetune(num_classes: int, pretrained=True, dropout=0.3):
    """ResNet-18 с разморозкой layer4 (fine-tuning) — как в чекпойнте 6."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    _freeze(model, trainable=False)
    for p in model.layer4.parameters():
        p.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def build_mobilenet_v2(num_classes: int, pretrained=True, dropout=0.2):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    _freeze(model, trainable=False)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def build_efficientnet_b0(num_classes: int, pretrained=True, unfreeze_last_blocks=True, dropout=0.3):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    _freeze(model, trainable=False)
    if unfreeze_last_blocks:
        for p in model.features[-2:].parameters():
            p.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def build_model(arch: str, num_classes: int, **kw) -> nn.Module:
    if arch == "vit_b_16":
        return build_vit_b16(
            num_classes,
            pretrained=kw.get("pretrained", True),
            unfreeze_last_blocks=kw.get("unfreeze_last_blocks", 2),
            dropout=kw.get("dropout", 0.3),
        )
    if arch == "resnet18":
        return build_resnet18(
            num_classes, pretrained=kw.get("pretrained", True), dropout=kw.get("dropout", 0.2)
        )
    if arch == "resnet18_finetune":
        return build_resnet18_finetune(
            num_classes, pretrained=kw.get("pretrained", True), dropout=kw.get("dropout", 0.3)
        )
    if arch == "mobilenet_v2":
        return build_mobilenet_v2(
            num_classes, pretrained=kw.get("pretrained", True), dropout=kw.get("dropout", 0.2)
        )
    if arch == "efficientnet_b0":
        return build_efficientnet_b0(
            num_classes, pretrained=kw.get("pretrained", True),
            unfreeze_last_blocks=kw.get("unfreeze_last_blocks", True), dropout=kw.get("dropout", 0.3)
        )
    raise ValueError(f"Unknown arch: {arch}")


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Обучение и оценка
# --------------------------------------------------------------------------- #
def _make_optimizer(name: str, params, lr: float, weight_decay: float = 0.0):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model.eval()
    all_logits, all_targets, running = [], [], 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running += loss.item() * images.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

    try:
        roc_auc = roc_auc_score(
            targets, probs, multi_class="ovr", average="macro",
            labels=list(range(num_classes)),
        )
    except ValueError:
        roc_auc = float("nan")

    return {
        "loss": running / len(loader.dataset),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
        "roc_auc_macro": float(roc_auc),
        "targets": targets,
        "preds": preds,
        "probs": probs,
    }


def fit(model, train_loader, val_loader, *, device, epochs, optimizer_name, lr,
        weight_decay, class_weights, num_classes, selection_metric="macro_f1"):
    """Полный цикл обучения с отбором лучшей эпохи по val-метрике."""
    import copy

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = _make_optimizer(optimizer_name, params, lr, weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [],
               "val_macro_f1": [], "val_roc_auc_macro": []}
    best_metric, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val = evaluate(model, val_loader, criterion, device, num_classes)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val["loss"])
        history["val_accuracy"].append(val["accuracy"])
        history["val_macro_f1"].append(val["macro_f1"])
        history["val_roc_auc_macro"].append(val["roc_auc_macro"])
        print(f"epoch {epoch:02d} | train_loss {tr_loss:.4f} | "
              f"val_f1 {val['macro_f1']:.4f} | val_acc {val['accuracy']:.4f} "
              f"| val_auc {val['roc_auc_macro']:.4f}")
        if val[selection_metric] > best_metric:
            best_metric = val[selection_metric]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_metric


# --------------------------------------------------------------------------- #
# Графики (сохраняются и логируются как артефакты)
# --------------------------------------------------------------------------- #
def plot_learning_curves(history: dict, out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["val_macro_f1"], label="val macro-F1")
    ax[1].plot(history["val_accuracy"], label="val accuracy")
    ax[1].set_title("Validation quality"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


def plot_confusion_matrix(targets, preds, classes, out_path: Path, normalize=True) -> Path:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    out_path = Path(out_path)
    cm = confusion_matrix(targets, preds, labels=list(range(len(classes))))
    title = "Confusion matrix"
    if normalize:
        cm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        title += " (normalized by true class / recall)"
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > (0.5 if normalize else cm.max() / 2) else "black",
                    fontsize=8)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Чекпойнт для сервиса (портативный, без зависимости от MLflow)
# --------------------------------------------------------------------------- #
def save_service_checkpoint(model: nn.Module, classes: List[str], config: dict, out_path: Path) -> Path:
    """Сохраняет state_dict + метаданные в формате, который читает FastAPI-сервис."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(classes),
            "arch": config.get("arch", "vit_b_16"),
            "config": config,
        },
        out_path,
    )
    return out_path
