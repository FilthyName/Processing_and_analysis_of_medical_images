"""
make_splits.py — пересплит ISIC-2018 / HAM10000 по lesion_id (Фаза 1).

Зачем: в HAM10000 у одного образования (lesion_id) бывает несколько снимков.
Если делить по картинке, снимки одного lesion попадают и в train, и в test —
это утечка, завышающая метрики. Здесь группируем по lesion_id, так что
каждое образование целиком оказывается ровно в одном сплите, и при этом
сохраняем стратификацию по классу.

Запуск:
    python src/make_splits.py \
        --metadata data/raw/HAM10000_metadata.csv \
        --images-dir data/raw/images \
        --out-dir data/splits \
        --old-splits-dir data/splits_old   # необязательно: померить старую утечку

Результат: data/splits/{train,val,test}.csv с колонками image,label,path,y,lesion_id
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Канонический порядок классов (как в service/artifacts/classes.json и conf/config.yaml)
CANON_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CANON_CLASSES)}


# --------------------------------------------------------------------------- #
# Построение полного датафрейма из метаданных
# --------------------------------------------------------------------------- #
def build_dataframe(metadata_csv: Path, images_dir: Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv)
    # HAM10000_metadata.csv: lesion_id, image_id, dx, dx_type, age, sex, localization
    required = {"lesion_id", "image_id", "dx"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"В метаданных нет колонок: {missing}. Это точно HAM10000_metadata.csv?")

    df = pd.DataFrame({
        "image": meta["image_id"].astype(str) + ".jpg",
        "label": meta["dx"].astype(str),
        "lesion_id": meta["lesion_id"].astype(str),
    })

    unknown = set(df["label"]) - set(CANON_CLASSES)
    if unknown:
        raise ValueError(f"Неизвестные классы в dx: {unknown}")

    df["y"] = df["label"].map(CLASS_TO_IDX).astype(int)
    df["path"] = df["image"].apply(lambda x: str(images_dir / x))

    # Санити: один lesion — один класс (в HAM10000 это так)
    multi = df.groupby("lesion_id")["label"].nunique()
    if (multi > 1).any():
        bad = multi[multi > 1].index.tolist()[:5]
        raise ValueError(f"Найдены lesion с разными классами (пример): {bad}")

    return df[["image", "label", "path", "y", "lesion_id"]]


# --------------------------------------------------------------------------- #
# Группо-стратифицированный сплит
# --------------------------------------------------------------------------- #
def _carve_one_fold(df: pd.DataFrame, frac: float, seed: int):
    """Отрезает ~frac данных одним фолдом StratifiedGroupKFold (группы не пересекаются)."""
    n_splits = max(2, round(1.0 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_idx, take_idx = next(sgkf.split(df, df["y"], groups=df["lesion_id"]))
    return df.iloc[keep_idx].reset_index(drop=True), df.iloc[take_idx].reset_index(drop=True)


def make_splits(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    # Стадия 1: отделяем test (его группы больше нигде не появятся)
    train_val, test = _carve_one_fold(df, test_size, seed)
    # Стадия 2: из остатка отделяем val. Доля val от остатка:
    val_frac_remaining = val_size / (1.0 - test_size)
    train, val = _carve_one_fold(train_val, val_frac_remaining, seed)
    return train, val, test


# --------------------------------------------------------------------------- #
# Проверки (шаг 1.2)
# --------------------------------------------------------------------------- #
def validate(train, val, test, rare_threshold=10):
    ok = True
    print("\n=== ПРОВЕРКА СПЛИТОВ ===")

    # 1) Непересечение групп
    g_tr, g_va, g_te = set(train.lesion_id), set(val.lesion_id), set(test.lesion_id)
    for a, b, na, nb in [(g_tr, g_va, "train", "val"), (g_tr, g_te, "train", "test"),
                         (g_va, g_te, "val", "test")]:
        inter = len(a & b)
        flag = "OK" if inter == 0 else "!!! УТЕЧКА"
        print(f"lesion_id overlap {na}∩{nb}: {inter}  [{flag}]")
        ok &= inter == 0

    # 2) Непересечение картинок (производное, но проверим)
    i_tr, i_va, i_te = set(train.image), set(val.image), set(test.image)
    assert not (i_tr & i_va) and not (i_tr & i_te) and not (i_va & i_te), "Пересечение image!"

    # 3) Размеры
    total_img = len(train) + len(val) + len(test)
    total_les = len(g_tr | g_va | g_te)
    print(f"\nИзображения: train {len(train)} / val {len(val)} / test {len(test)} "
          f"= {total_img}")
    print(f"Lesion:      train {len(g_tr)} / val {len(g_va)} / test {len(g_te)} "
          f"= {total_les}")
    print(f"Доли (img):  train {len(train)/total_img:.3f} / "
          f"val {len(val)/total_img:.3f} / test {len(test)/total_img:.3f}")

    # 4) Распределение классов
    dist = pd.DataFrame({
        "train": train.label.value_counts(normalize=True).round(3),
        "val": val.label.value_counts(normalize=True).round(3),
        "test": test.label.value_counts(normalize=True).round(3),
    }).reindex(CANON_CLASSES)
    print("\nДоли классов по сплитам:")
    print(dist.to_string())

    # 5) Редкие классы в val/test
    print("\nАбсолютные counts (важно для df/vasc):")
    counts = pd.DataFrame({
        "train": train.label.value_counts(),
        "val": val.label.value_counts(),
        "test": test.label.value_counts(),
    }).reindex(CANON_CLASSES).fillna(0).astype(int)
    print(counts.to_string())
    for split_name in ["val", "test"]:
        low = counts[counts[split_name] < rare_threshold][split_name]
        for cls, n in low.items():
            print(f"  ВНИМАНИЕ: класс {cls} в {split_name} = {n} (< {rare_threshold}) — "
                  f"метрика по нему будет шумной")

    print("\nИтог:", "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ" if ok else "ЕСТЬ ПРОБЛЕМЫ — не использовать!")
    return ok


def measure_old_leakage(old_dir: Path, df: pd.DataFrame):
    """Сколько lesion в старых сплитах были размазаны по нескольким частям (= утечка)."""
    files = {n: old_dir / f"{n}.csv" for n in ["train", "val", "test"]}
    if not all(p.exists() for p in files.values()):
        print(f"\n(Старые сплиты в {old_dir} не найдены — диагностику утечки пропускаю)")
        return
    img2les = dict(zip(df.image, df.lesion_id))
    membership = {}  # lesion_id -> set(split)
    for split_name, p in files.items():
        old = pd.read_csv(p)
        for img in old["image"].astype(str):
            les = img2les.get(img)
            if les is not None:
                membership.setdefault(les, set()).add(split_name)
    leaked = {l: s for l, s in membership.items() if len(s) > 1}
    total = len(membership)
    print("\n=== ДИАГНОСТИКА СТАРЫХ СПЛИТОВ (для защиты) ===")
    print(f"Lesion, размазанных по >1 сплиту: {len(leaked)} из {total} "
          f"({100*len(leaked)/max(total,1):.1f}%)")
    print("Именно это завышало метрики до пересплита.")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, type=Path,
                    help="путь к HAM10000_metadata.csv")
    ap.add_argument("--images-dir", default=Path("data/raw/images"), type=Path)
    ap.add_argument("--out-dir", default=Path("data/splits"), type=Path)
    ap.add_argument("--old-splits-dir", default=None, type=Path,
                    help="старые сплиты — чтобы померить прежнюю утечку")
    ap.add_argument("--test-size", default=0.15, type=float)
    ap.add_argument("--val-size", default=0.15, type=float)
    ap.add_argument("--rare-threshold", default=10, type=int)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    df = build_dataframe(args.metadata, args.images_dir)
    print(f"Всего: {len(df)} изображений, {df.lesion_id.nunique()} уникальных lesion")

    train, val, test = make_splits(df, args.test_size, args.val_size, args.seed)
    ok = validate(train, val, test, rare_threshold=args.rare_threshold)

    if args.old_splits_dir:
        measure_old_leakage(args.old_splits_dir, df)

    if not ok:
        raise SystemExit("Сплиты не прошли проверку — файлы НЕ сохранены.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        out = args.out_dir / f"{name}.csv"
        part.to_csv(out, index=False)
        print(f"Сохранено: {out} ({len(part)} строк)")
    print(f"\nСплиты по lesion_id готовы. Seed={args.seed} зафиксирован.")


if __name__ == "__main__":
    main()
