from __future__ import annotations
import base64
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import logging
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from PIL import Image
from torchvision import models, transforms
import numpy as np
from service.database import SessionLocal, RequestHistory, init_db

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("service/artifacts")
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "model.pth"
CLASSES_JSON_PATH = ARTIFACTS_DIR / "classes.json"
PREPROC_CONFIG_PATH = ARTIFACTS_DIR / "preprocessing_config.json"

_model: Optional[torch.nn.Module] = None
_classes: Optional[List[str]] = None
_preprocess: Optional[transforms.Compose] = None
MEL_THRESHOLD = 0.14


def _load_classes() -> List[str]:
    if not CLASSES_JSON_PATH.exists():
        raise FileNotFoundError(f"classes.json не найден: {CLASSES_JSON_PATH}")
    with open(CLASSES_JSON_PATH, "r") as f:
        classes = json.load(f)
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
        raise ValueError("classes.json должен содержать список строк")
    return classes


def _load_preprocess() -> transforms.Compose:
    defaults = {
        "img_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }
    if PREPROC_CONFIG_PATH.exists():
        with open(PREPROC_CONFIG_PATH, "r") as f:
            cfg = {**defaults, **json.load(f)}
        logger.info("Загружен preprocessing_config.json")
    else:
        cfg = defaults
        logger.warning(
            "preprocessing_config.json не найден, используем дефолтные значения ImageNet"
        )

    return transforms.Compose(
        [
            transforms.Resize((cfg["img_size"], cfg["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["normalize_mean"], std=cfg["normalize_std"]),
        ]
    )


def _load_model(model_path: Path, num_classes: int) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"model.pth не найден: {model_path}")

    ckpt = torch.load(str(model_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(
            "Неожиданный формат чекпоинта (ожидается dict с ключом 'state_dict')"
        )

    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features, num_classes),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


app = FastAPI(title="Skin Lesion Classifier — ViT-B/16", version="2.0.0")


@app.on_event("startup")
def _startup() -> None:
    global _model, _classes, _preprocess
    _classes = _load_classes()
    _preprocess = _load_preprocess()
    _model = _load_model(DEFAULT_MODEL_PATH, num_classes=len(_classes))
    init_db()
    logger.info("Сервис запущен. Классы: %s", _classes)


def _bad_request(detail: str = "bad request") -> PlainTextResponse:
    return PlainTextResponse(detail, status_code=400)


def _model_failed() -> PlainTextResponse:
    return PlainTextResponse("модель не смогла обработать данные", status_code=403)


def _pil_from_bytes(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    return img.convert("RGB") if img.mode != "RGB" else img


def _apply_mel_threshold(probs: np.ndarray) -> int:
    assert _classes is not None
    if "mel" in _classes:
        mel_idx = _classes.index("mel")
        if probs[mel_idx] >= MEL_THRESHOLD:
            return mel_idx
    return int(probs.argmax())


def _probs_to_result(probs: np.ndarray, top_k: int) -> Dict[str, Any]:
    assert _classes is not None
    pred_idx = _apply_mel_threshold(probs)
    pred_class = _classes[pred_idx]
    confidence = float(probs[pred_idx])
    top_k = max(1, min(int(top_k), len(_classes)))
    top_indices = probs.argsort()[::-1][:top_k].tolist()
    top = [{"class": _classes[i], "probability": float(probs[i])} for i in top_indices]
    return {
        "predicted_class": pred_class,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "top_k": top,
        "probs": {cls: float(p) for cls, p in zip(_classes, probs)},
    }


def _predict_global(image_bytes: bytes, top_k: int = 3) -> Dict[str, Any]:
    assert _model is not None and _preprocess is not None
    img = _pil_from_bytes(image_bytes)
    x = _preprocess(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_model(x), dim=1).squeeze(0).cpu().numpy()
    return _probs_to_result(probs, top_k)


def _predict_windows(
    image_bytes: bytes,
    top_k: int = 3,
    window_size: int = 224,
    stride: int = 112,
) -> Dict[str, Any]:
    assert _model is not None and _preprocess is not None
    img = _pil_from_bytes(image_bytes)
    w, h = img.size
    all_probs: list = []

    for y in range(0, max(1, h - window_size + 1), stride):
        for x in range(0, max(1, w - window_size + 1), stride):
            crop = img.crop((x, y, x + window_size, y + window_size))
            tensor = _preprocess(crop).unsqueeze(0)
            with torch.no_grad():
                p = torch.softmax(_model(tensor), dim=1).squeeze(0).cpu().numpy()
            all_probs.append(p)

    if not all_probs:
        return _predict_global(image_bytes, top_k)

    return _probs_to_result(np.mean(all_probs, axis=0), top_k)


@app.get("/model-info")
def model_info():
    return {
        "model_name": "skin-lesion-classifier",
        "architecture": "ViT-B/16",
        "stage": "PRD",
        "classes": _classes or [],
        "inference_modes": ["global", "windows"],
    }


@app.post("/forward", response_model=None)
async def forward(request: Request) -> Response:
    headers = request.headers

    try:
        top_k = int(headers.get("x-top-k", "3"))
    except Exception:
        return _bad_request("x-top-k должен быть числом")

    mode = headers.get("x-mode", "global").strip().lower()
    if mode not in {"global", "windows"}:
        return _bad_request("x-mode должен быть 'global' или 'windows'")

    return_probs = headers.get("x-return-probs", "true").strip().lower() in {
        "true",
        "1",
        "yes",
    }
    return_image = headers.get("x-return-image", "true").strip().lower() in {
        "true",
        "1",
        "yes",
    }

    content_type = (headers.get("content-type") or "").lower()

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            if "image" not in form:
                return _bad_request("Поле 'image' не найдено в форме")
            upload = form["image"]
            image_bytes = (
                await upload.read() if hasattr(upload, "read") else bytes(upload)
            )

        elif "application/json" in content_type:
            payload = await request.json()
            if not isinstance(payload, dict) or "image_b64" not in payload:
                return _bad_request("Ожидается JSON с полем 'image_b64'")
            image_bytes = base64.b64decode(payload["image_b64"], validate=True)
        else:
            return _bad_request(
                "content-type должен быть multipart/form-data или application/json"
            )

        try:
            pil = _pil_from_bytes(image_bytes)
            image_w, image_h = pil.size
        except Exception:
            image_w, image_h = None, None

        t0 = time.perf_counter()
        pred = (
            _predict_windows(image_bytes, top_k)
            if mode == "windows"
            else _predict_global(image_bytes, top_k)
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        try:
            db = SessionLocal()
            top3 = pred["top_k"][:3]
            db.add(
                RequestHistory(
                    elapsed_ms=float(elapsed_ms),
                    image_width=int(image_w) if image_w is not None else None,
                    image_height=int(image_h) if image_h is not None else None,
                    predicted_class=pred["predicted_class"],
                    confidence=pred["confidence"],
                    top3_classes=",".join(t["class"] for t in top3),
                    top3_probs=",".join(str(round(t["probability"], 4)) for t in top3),
                    mode=mode,
                    model_name="skin-lesion-classifier",
                    architecture="ViT-B/16",
                    stage="PRD",
                )
            )
            db.commit()
        except Exception as e:
            logger.warning("Ошибка записи в БД: %s", e)
        finally:
            db.close()

        response: Dict[str, Any] = {
            "predicted_class": pred["predicted_class"],
            "confidence": pred["confidence"],
            "top_k": pred["top_k"],
            "mode": mode,
            "elapsed_ms": float(elapsed_ms),
        }
        if return_probs:
            response["probs"] = pred["probs"]
        if return_image:
            response["image_b64"] = base64.b64encode(image_bytes).decode("ascii")

        return JSONResponse(response, status_code=200)

    except Exception:
        logger.exception("Ошибка инференса в /forward")
        return _model_failed()


@app.get("/history")
def get_history(limit: int = 100, offset: int = 0):
    try:
        db = SessionLocal()
        rows = (
            db.query(RequestHistory)
            .order_by(RequestHistory.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        db.close()
    except Exception:
        logger.exception("DB error in /history")
        raise HTTPException(status_code=500, detail="database error")

    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "elapsed_ms": r.elapsed_ms,
            "image_size": [r.image_width, r.image_height],
            "predicted_class": r.predicted_class,
            "confidence": r.confidence,
            "mode": r.mode,
            "architecture": r.architecture,
            "stage": r.stage,
        }
        for r in rows
    ]


@app.get("/stats")
def stats():
    db = SessionLocal()
    try:
        rows = db.query(RequestHistory).all()
        times = [r.elapsed_ms for r in rows if r.elapsed_ms is not None]
        if not times:
            return JSONResponse(status_code=204, content={})
        arr = np.array(times)
        return {
            "count": int(len(arr)),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
        }
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok"}
