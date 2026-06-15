"""
7ой чекпоинт

Запускается из CLI с Hydra-конфигом:
    python src/dl_experiments.py
    python src/dl_experiments.py train.epochs=15 model.lr=2e-5 robustness.enabled=false

Что делает:
  1. Фиксирует seed и среду  -> воспроизводимость.
  2. Обучает baseline (ResNet-18 frozen) и логирует отдельным MLflow-раном.
  3. Переобучает финальную модель (ViT-B/16) и логирует параметры/метрики/артефакты.
  4. Сохраняет артефакты в S3/MinIO: модель, confusion matrix, learning curves,
     примеры предсказаний.
  5. Регистрирует финальную модель в Model Registry и вешает алиас PRD.
  6. Проводит структурный анализ ошибок (10–20 примеров).
  7. Сравнивает финальную модель с baseline.
  8. Проверяет robustness на возмущённых входах.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import hydra
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

import common as C
import threshold_tuning as TT


# --------------------------------------------------------------------------- #
# Анализ ошибок
# --------------------------------------------------------------------------- #
def error_analysis(eval_out, test_df, classes, images_dir, out_dir: Path, num_examples=16):
    """Выделяет типичные ошибки, рисует grid примеров и считает наиболее путаемые пары."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from sklearn.metrics import confusion_matrix

    out_dir = Path(out_dir)
    targets, preds, probs = eval_out["targets"], eval_out["preds"], eval_out["probs"]
    test_df = test_df.reset_index(drop=True)

    wrong_idx = np.where(preds != targets)[0]
    conf = probs[np.arange(len(preds)), preds]
    # самые уверенные ошибки — самые показательные для разбора
    wrong_sorted = wrong_idx[np.argsort(-conf[wrong_idx])]
    take = wrong_sorted[:num_examples]

    rows, errors_table = [], []
    cols = 4
    n = len(take)
    rows_n = int(np.ceil(n / cols)) if n else 1
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3, rows_n * 3))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for k, i in enumerate(take):
        row = test_df.iloc[i]
        img = Image.open(C._remap_path(row["path"], Path(images_dir))).convert("RGB")
        axes[k].imshow(img)
        axes[k].set_title(f"true={classes[targets[i]]}\npred={classes[preds[i]]} ({conf[i]:.2f})",
                          fontsize=8)
        errors_table.append({
            "image": row["image"], "true": classes[targets[i]],
            "pred": classes[preds[i]], "pred_prob": float(conf[i]),
        })
    fig.tight_layout()
    grid_path = out_dir / "error_examples.png"
    fig.savefig(grid_path, dpi=150); plt.close(fig)

    # Наиболее путаемые пары классов (по нормированной CM)
    cm = confusion_matrix(targets, preds, labels=list(range(len(classes)))).astype(float)
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                pairs.append((classes[i], classes[j], int(cm[i, j]), float(cm_norm[i, j])))
    pairs.sort(key=lambda x: -x[2])

    pd.DataFrame(errors_table).to_csv(out_dir / "error_examples.csv", index=False)
    confused = pd.DataFrame(pairs, columns=["true", "pred", "count", "rate_of_true_class"])
    confused.to_csv(out_dir / "confused_pairs.csv", index=False)
    print("\nТОП путаемых пар (true -> pred):")
    print(confused.head(8).to_string(index=False))
    return grid_path, out_dir / "error_examples.csv", out_dir / "confused_pairs.csv"


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
def robustness_check(model, test_df, cfg, classes, device, out_dir: Path):
    """Возмущает входы и измеряет долю изменившихся предсказаний (prediction flip rate)."""
    import io
    from PIL import Image
    from torchvision import transforms

    out_dir = Path(out_dir)
    images_dir = Path(cfg.data.images_dir)
    size = cfg.data.image_size
    normalize = transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)
    base_tf = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor(), normalize]
    )

    def perturb(img: Image.Image, kind: str) -> Image.Image:
        if kind == "hflip":
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        if kind == "rotate10":
            return img.rotate(10)
        if kind == "brightness":
            from PIL import ImageEnhance
            return ImageEnhance.Brightness(img).enhance(1.3)
        if kind == "gaussian_noise":
            arr = np.array(img).astype(np.float32)
            arr += np.random.normal(0, 12, arr.shape)
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if kind == "jpeg_compression":
            buf = io.BytesIO(); img.save(buf, "JPEG", quality=40); buf.seek(0)
            return Image.open(buf).convert("RGB")
        return img

    @torch.no_grad()
    def predict(img: Image.Image) -> int:
        x = base_tf(img).unsqueeze(0).to(device)
        return int(model(x).argmax(dim=1).item())

    sample = test_df.sample(n=min(cfg.robustness.sample_size, len(test_df)),
                            random_state=cfg.seed).reset_index(drop=True)
    model.eval()
    results = {}
    for kind in cfg.robustness.perturbations:
        flips = 0
        for _, row in sample.iterrows():
            img = Image.open(C._remap_path(row["path"], images_dir)).convert("RGB")
            base_pred = predict(img)
            pert_pred = predict(perturb(img, kind))
            flips += int(base_pred != pert_pred)
        rate = flips / len(sample)
        results[kind] = round(rate, 4)
        print(f"robustness | {kind:18s} prediction flip rate = {rate:.3f}")

    (out_dir / "robustness.json").write_text(json.dumps(results, indent=2))
    return results, out_dir / "robustness.json"


# --------------------------------------------------------------------------- #
# Один обучающий ран -> MLflow
# --------------------------------------------------------------------------- #
def run_training(cfg, *, arch, epochs, lr, optimizer, augmentation, run_name,
                 classes, device, work_dir: Path, tag_prd=False, do_extras=False):
    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    train_cfg.train.augmentation = augmentation
    train_cfg.train.batch_size = cfg.train.batch_size if arch == "vit_b_16" else 32

    train_loader, val_loader, test_loader, (train_df, val_df, test_df) = C.make_loaders(train_cfg, classes)
    class_weights = C.compute_class_weights(train_df, classes) if cfg.train.use_class_weights else None

    model = C.build_model(
        arch, num_classes=len(classes), pretrained=cfg.model.pretrained,
        unfreeze_last_blocks=cfg.model.unfreeze_last_blocks, dropout=cfg.model.dropout,
    )

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "arch": arch, "pretrained": cfg.model.pretrained, "epochs": epochs,
            "lr": lr, "optimizer": optimizer, "batch_size": train_cfg.train.batch_size,
            "augmentation": augmentation, "dropout": cfg.model.dropout,
            "unfreeze_last_blocks": cfg.model.unfreeze_last_blocks,
            "weight_decay": cfg.model.weight_decay, "use_class_weights": cfg.train.use_class_weights,
            "seed": cfg.seed, "image_size": cfg.data.image_size,
            "selection_metric": cfg.train.selection_metric, "num_classes": len(classes),
            "dataset": "ISIC-2018 / HAM10000 (7 classes)",
        })
        mlflow.set_tag("trainable_params", C.count_trainable_params(model))

        t0 = time.time()
        model, history, best_val = C.fit(
            model, train_loader, val_loader, device=device, epochs=epochs,
            optimizer_name=optimizer, lr=lr, weight_decay=cfg.model.weight_decay,
            class_weights=class_weights, num_classes=len(classes),
            selection_metric=cfg.train.selection_metric,
        )
        train_minutes = (time.time() - t0) / 60.0

        # --- метрики train/val/test ---
        import torch.nn as nn
        crit = nn.CrossEntropyLoss()
        val_out = C.evaluate(model, val_loader, crit, device, len(classes))
        test_out = C.evaluate(model, test_loader, crit, device, len(classes))
        for split, out in [("val", val_out), ("test", test_out)]:
            mlflow.log_metrics({
                f"{split}_accuracy": out["accuracy"],
                f"{split}_macro_f1": out["macro_f1"],
                f"{split}_roc_auc_macro": out["roc_auc_macro"],
            })
        mlflow.log_metric("train_minutes", train_minutes)
        for ep, (a, f1) in enumerate(zip(history["val_accuracy"], history["val_macro_f1"]), 1):
            mlflow.log_metric("val_macro_f1_curve", f1, step=ep)

        # --- артефакты: графики ---
        work_dir.mkdir(parents=True, exist_ok=True)
        lc = C.plot_learning_curves(history, work_dir / "learning_curves.png")
        cm = C.plot_confusion_matrix(test_out["targets"], test_out["preds"], classes,
                                     work_dir / "confusion_matrix.png", normalize=True)
        mlflow.log_artifact(str(lc), artifact_path="plots")
        mlflow.log_artifact(str(cm), artifact_path="plots")

        # --- classification report ---
        from sklearn.metrics import classification_report
        report = classification_report(test_out["targets"], test_out["preds"],
                                       target_names=classes, output_dict=False)
        (work_dir / "classification_report.txt").write_text(report)
        mlflow.log_artifact(str(work_dir / "classification_report.txt"), artifact_path="reports")

        # --- порог меланомы (только для финальной модели) ---
        mel_threshold = None
        if do_extras and cfg.mel_threshold.enabled:
            mel_threshold = TT.choose_threshold(
                val_out["targets"], val_out["probs"], classes,
                positive=cfg.mel_threshold.positive,
                target_sensitivity=cfg.mel_threshold.target_sensitivity,
            )
            mlflow.log_param("mel_threshold", mel_threshold)
            mlflow.log_param("mel_threshold_target_sensitivity", cfg.mel_threshold.target_sensitivity)
            print(f"Подобран порог mel = {mel_threshold} (target sens {cfg.mel_threshold.target_sensitivity})")

        # --- сохранение модели ---
        mlflow.pytorch.log_model(model, name="model")
        ckpt_config = {"arch": arch, "dropout": cfg.model.dropout,
                       "unfreeze_last_blocks": cfg.model.unfreeze_last_blocks}
        if mel_threshold is not None:
            ckpt_config["mel_threshold"] = mel_threshold
            ckpt_config["mel_index"] = classes.index(cfg.mel_threshold.positive)
        svc_ckpt = C.save_service_checkpoint(model, classes, ckpt_config, work_dir / "model.pth")
        mlflow.log_artifact(str(svc_ckpt), artifact_path="service_checkpoint")

        result = {
            "run_id": run.info.run_id, "arch": arch,
            "test_accuracy": test_out["accuracy"], "test_macro_f1": test_out["macro_f1"],
            "test_roc_auc_macro": test_out["roc_auc_macro"], "train_minutes": train_minutes,
        }

        # --- доп. блок только для финальной модели ---
        if do_extras:
            grid, _, _ = error_analysis(
                test_out, test_df, classes, cfg.data.images_dir, work_dir,
                num_examples=cfg.error_analysis.num_examples,
            )
            mlflow.log_artifact(str(grid), artifact_path="error_analysis")
            mlflow.log_artifact(str(work_dir / "error_examples.csv"), artifact_path="error_analysis")
            mlflow.log_artifact(str(work_dir / "confused_pairs.csv"), artifact_path="error_analysis")
            if cfg.robustness.enabled:
                rob, rob_path = robustness_check(model, test_df, cfg, classes, device, work_dir)
                mlflow.log_metrics({f"robust_flip_{k}": v for k, v in rob.items()})
                mlflow.log_artifact(str(rob_path), artifact_path="robustness")
            # порог mel: метрики до/после + график trade-off
            if mel_threshold is not None:
                comp = TT.compare(test_out["targets"], test_out["probs"], classes,
                                  mel_threshold, positive=cfg.mel_threshold.positive)
                comp.to_csv(work_dir / "mel_threshold_compare.csv", index=False)
                tradeoff = TT.plot_tradeoff(test_out["targets"], test_out["probs"], classes,
                                            work_dir / "mel_threshold_tradeoff.png",
                                            positive=cfg.mel_threshold.positive)
                row = comp.iloc[-1]  # вариант с порогом
                mlflow.log_metrics({
                    "test_mel_sensitivity_thr": float(row[f"{cfg.mel_threshold.positive}_sensitivity"]),
                    "test_mel_precision_thr": float(row[f"{cfg.mel_threshold.positive}_precision"]),
                })
                mlflow.log_artifact(str(work_dir / "mel_threshold_compare.csv"), artifact_path="mel_threshold")
                mlflow.log_artifact(str(tradeoff), artifact_path="mel_threshold")

        # --- регистрация PRD ---
        if tag_prd:
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, cfg.mlflow.registered_model_name)
            client = mlflow.tracking.MlflowClient()
            # современный способ: alias; плюс legacy-тег для совместимости
            try:
                client.set_registered_model_alias(
                    cfg.mlflow.registered_model_name, cfg.mlflow.prod_alias, mv.version)
            except Exception as e:
                print("alias set failed (старый MLflow?):", e)
            client.set_model_version_tag(
                cfg.mlflow.registered_model_name, mv.version, "stage", cfg.mlflow.prod_alias)
            if mel_threshold is not None:
                client.set_model_version_tag(
                    cfg.mlflow.registered_model_name, mv.version, "mel_threshold", str(mel_threshold))
                client.set_model_version_tag(
                    cfg.mlflow.registered_model_name, mv.version, "mel_index",
                    str(classes.index(cfg.mel_threshold.positive)))
            print(f"Зарегистрирована версия {mv.version} с алиасом/тегом {cfg.mlflow.prod_alias}")
            result["registered_version"] = mv.version

    return result


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    C.set_seed(cfg.seed)
    device = C.resolve_device(cfg.device)
    classes = list(cfg.data.classes)
    work_root = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    summary = {}

    # 1) Baseline
    if cfg.baseline.enabled:
        print("\n=== BASELINE (ResNet-18 frozen) ===")
        summary["baseline"] = run_training(
            cfg, arch=cfg.baseline.arch, epochs=cfg.baseline.epochs, lr=cfg.baseline.lr,
            optimizer=cfg.baseline.optimizer, augmentation=cfg.baseline.augmentation,
            run_name="baseline_resnet18", classes=classes, device=device,
            work_dir=work_root / "baseline", tag_prd=False, do_extras=False,
        )

    # 2) Финальная модель (ViT-B/16) + PRD + анализ ошибок + robustness
    print("\n=== FINAL (ViT-B/16) ===")
    summary["final"] = run_training(
        cfg, arch=cfg.model.arch, epochs=cfg.train.epochs, lr=cfg.model.lr,
        optimizer=cfg.model.optimizer, augmentation=cfg.train.augmentation,
        run_name=cfg.mlflow.run_name, classes=classes, device=device,
        work_dir=work_root / "final", tag_prd=True, do_extras=True,
    )

    # 3) Сравнение с baseline
    if "baseline" in summary:
        b, f = summary["baseline"], summary["final"]
        print("\n=== СРАВНЕНИЕ С BASELINE (test) ===")
        comp = pd.DataFrame([
            {"model": "baseline ResNet-18", "accuracy": b["test_accuracy"],
             "macro_f1": b["test_macro_f1"], "roc_auc_macro": b["test_roc_auc_macro"]},
            {"model": "final ViT-B/16", "accuracy": f["test_accuracy"],
             "macro_f1": f["test_macro_f1"], "roc_auc_macro": f["test_roc_auc_macro"]},
        ])
        print(comp.to_string(index=False))
        print(f"\nΔ macro-F1 (final - baseline) = "
              f"{f['test_macro_f1'] - b['test_macro_f1']:+.4f}")

    (work_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nГотово. Сводка: {work_root / 'summary.json'}")
    print("Открой MLflow UI:", cfg.mlflow.tracking_uri)


if __name__ == "__main__":
    main()
