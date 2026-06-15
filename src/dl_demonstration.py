from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import mlflow.pytorch
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

import common as C


def load_prd_model(cfg):
    """Сначала пробуем alias (новый MLflow), затем fallback по тегу stage=PRD.
    Возвращает (модель, порог_mel или None, индекс_mel)."""
    name, alias = cfg.mlflow.registered_model_name, cfg.mlflow.prod_alias
    client = mlflow.tracking.MlflowClient()

    def _threshold_from_version(version):
        try:
            mv = client.get_model_version(name, version)
            thr = mv.tags.get("mel_threshold")
            idx = mv.tags.get("mel_index")
            return (float(thr) if thr is not None else None,
                    int(idx) if idx is not None else None)
        except Exception:
            return None, None

    try:
        model = mlflow.pytorch.load_model(f"models:/{name}@{alias}")
        ver = client.get_model_version_by_alias(name, alias).version
        thr, idx = _threshold_from_version(ver)
        return model, thr, idx
    except Exception as e:
        print(f"alias-загрузка не удалась ({e}); ищу версию по тегу stage={alias}")
        for mv in client.search_model_versions(f"name='{name}'"):
            if mv.tags.get("stage") == alias:
                model = mlflow.pytorch.load_model(f"models:/{name}/{mv.version}")
                thr, idx = _threshold_from_version(mv.version)
                return model, thr, idx
        raise RuntimeError(f"Не найдена версия модели {name} с PRD-алиасом/тегом")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    C.set_seed(cfg.seed)
    device = C.resolve_device(cfg.device)
    classes = list(cfg.data.classes)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    model, mel_threshold, mel_idx = load_prd_model(cfg)
    model = model.to(device).eval()
    msg = f"PRD-модель загружена ({cfg.mlflow.registered_model_name}@{cfg.mlflow.prod_alias})"
    if mel_threshold is not None:
        msg += f" | порог mel={mel_threshold} (idx {mel_idx})"
    print(msg)

    size = cfg.data.image_size
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])

    image_path = cfg.get("demo", {}).get("image") if "demo" in cfg else None
    if not image_path:
        # дефолт — первый файл из тестового сплита
        _, _, test_df = C.load_splits(cfg.data.splits_dir)
        image_path = C._remap_path(test_df.iloc[0]["path"], Path(cfg.data.images_dir))
        print("demo.image не задан — беру первый пример из test:", image_path)

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()

    # правило с порогом меланомы (если он сохранён с моделью)
    pred_idx = int(probs.argmax())
    if mel_threshold is not None and mel_idx is not None and probs[mel_idx] >= mel_threshold:
        pred_idx = mel_idx
        print(f"(порог сработал: P(mel)={probs[mel_idx]:.3f} >= {mel_threshold})")

    top_k = int(cfg.get("demo", {}).get("top_k", 3)) if "demo" in cfg else 3
    order = probs.argsort()[::-1][:top_k]
    print(f"\nПредсказание для {Path(str(image_path)).name}:")
    print(f"  -> {classes[pred_idx]}  (p={probs[pred_idx]:.3f})")
    print("  top-k по вероятности:")
    for i in order:
        print(f"     {classes[i]:8s} {probs[i]:.3f}")


if __name__ == "__main__":
    main()
