from __future__ import annotations
import base64
import io
import time
from typing import Any, Dict, Optional, Tuple
import torch
import logging
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
from torchvision import models, transforms
import numpy as np
from service.database import (
    SessionLocal,
    RequestHistory,
    init_db,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "service/artifacts/model.pth"
IMG_SIZE = 224
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

app = FastAPI(title="ML Service (FastAPI) - /forward", version="1.0.0")
_model: Optional[torch.nn.Module] = None
_classes: Optional[list[str]] = None
_preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ]
)


def _bad_request() -> PlainTextResponse:
    return PlainTextResponse("bad request", status_code=400)


def _model_failed() -> PlainTextResponse:
    return PlainTextResponse("модель не смогла обработать данные", status_code=403)


def _load_checkpoint(model_path: str) -> Tuple[torch.nn.Module, list[str]]:
    ckpt = torch.load(model_path, map_location="cpu")

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "classes" not in ckpt:
        raise ValueError("Unexpected checkpoint format")

    classes = ckpt["classes"]

    if not isinstance(classes, list) or not all(isinstance(x, str) for x in classes):
        raise ValueError("Invalid classes in checkpoint")

    num_classes = len(classes)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


def _pil_from_bytes(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def _predict(image_bytes: bytes, top_k: int = 3) -> Dict[str, Any]:
    assert _model is not None
    assert _classes is not None
    img = _pil_from_bytes(image_bytes)
    x = _preprocess(img).unsqueeze(0)

    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_class = _classes[pred_idx]
    top_k = max(1, min(int(top_k), len(_classes)))
    top_indices = probs.argsort()[::-1][:top_k].tolist()
    top = [{"class": _classes[i], "prob": float(probs[i])} for i in top_indices]
    return {
        "predicted_index": pred_idx,
        "predicted_class": pred_class,
        "top_k": top,
        "probs": {cls: float(p) for cls, p in zip(_classes, probs)},
    }


@app.on_event("startup")
def _startup() -> None:
    global _model, _classes
    _model, _classes = _load_checkpoint(DEFAULT_MODEL_PATH)
    init_db()


@app.post("/forward", response_model=None)
async def forward(request: Request) -> Response:
    headers = request.headers

    try:
        top_k = int(headers.get("x-top-k", "3"))
    except Exception:
        return _bad_request()
    return_probs_raw = headers.get("x-return-probs", "true").strip().lower()

    if return_probs_raw not in {"true", "false", "1", "0", "yes", "no"}:
        return _bad_request()

    return_probs = return_probs_raw in {"true", "1", "yes"}
    return_image_raw = headers.get("x-return-image", "true").strip().lower()

    if return_image_raw not in {"true", "false", "1", "0", "yes", "no"}:
        return _bad_request()

    return_image = return_image_raw in {"true", "1", "yes"}
    content_type = (headers.get("content-type") or "").lower()

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()

            if "image" not in form:
                return _bad_request()

            upload = form["image"]

            if hasattr(upload, "read"):
                image_bytes = await upload.read()
            elif isinstance(upload, (bytes, bytearray)):
                image_bytes = bytes(upload)
            else:
                return _bad_request()

        elif "application/json" in content_type:
            payload = await request.json()

            if not isinstance(payload, dict) or "image_b64" not in payload:
                return _bad_request()

            try:
                image_bytes = base64.b64decode(payload["image_b64"], validate=True)
            except Exception:
                return _bad_request()
        else:
            return _bad_request()

        try:
            pil = _pil_from_bytes(image_bytes)
            image_w, image_h = pil.size
        except Exception:
            image_w, image_h = None, None

        started = time.perf_counter()
        pred = _predict(image_bytes=image_bytes, top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        try:
            db = SessionLocal()
            try:
                db.add(
                    RequestHistory(
                        elapsed_ms=float(elapsed_ms),
                        image_width=int(image_w) if image_w is not None else None,
                        image_height=int(image_h) if image_h is not None else None,
                        predicted_class=pred["predicted_class"],
                    )
                )
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.warning("Failed to store history entry (forward): %s", e)

        response = {
            "elapsed_ms": float(elapsed_ms),
            "predicted_class": pred["predicted_class"],
            "predicted_index": pred["predicted_index"],
            "top_k": pred["top_k"],
        }

        if return_probs:
            response["probs"] = pred["probs"]

        if return_image:
            response["image_b64"] = base64.b64encode(image_bytes).decode("ascii")

        return JSONResponse(response, status_code=200)

    except Exception as exc:
        try:
            db = SessionLocal()
            try:
                db.add(
                    RequestHistory(
                        elapsed_ms=None,
                        image_width=None,
                        image_height=None,
                        predicted_class=None,
                    )
                )
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.warning("Failed to store error history entry (forward): %s", e)

        logger.exception("Inference failed in /forward")
        return _model_failed()


@app.get("/history")
def get_history(limit: int = 100, offset: int = 0):
    try:
        db = SessionLocal()
        try:
            rows = (
                db.query(RequestHistory)
                .order_by(RequestHistory.timestamp.desc())
                .offset(int(offset))
                .limit(int(limit))
                .all()
            )
        finally:
            db.close()
    except Exception as e:
        logger.exception("DB error in get_history")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="database error"
        )

    if not rows:
        return []

    out = []
    for r in rows:
        out.append(
            {
                "id": int(getattr(r, "id", 0)),
                "timestamp": (
                    getattr(r, "timestamp").isoformat()
                    if getattr(r, "timestamp", None)
                    else None
                ),
                "elapsed_ms": (
                    float(getattr(r, "elapsed_ms", None))
                    if getattr(r, "elapsed_ms", None) is not None
                    else None
                ),
                "image_size": [
                    (
                        int(getattr(r, "image_width"))
                        if getattr(r, "image_width", None) is not None
                        else None
                    ),
                    (
                        int(getattr(r, "image_height"))
                        if getattr(r, "image_height", None) is not None
                        else None
                    ),
                ],
                "predicted_class": getattr(r, "predicted_class", None),
            }
        )
    return out


@app.get("/stats")
def stats():
    db = SessionLocal()
    try:
        rows = db.query(RequestHistory).all()

        times = [r.elapsed_ms for r in rows if r.elapsed_ms is not None]
        if not times:
            return JSONResponse(status_code=204, content={})

        arr = np.array(times)
        time_stats = {
            "count": int(len(arr)),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
        }

        widths = [r.image_width for r in rows if r.image_width is not None]
        heights = [r.image_height for r in rows if r.image_height is not None]

        def size_stats(values):
            if not values:
                return {}
            a = np.array(values, dtype=float)
            return {
                "count": int(a.size),
                "min": int(a.min()),
                "max": int(a.max()),
                "mean": float(a.mean()),
            }

        iw_stats = size_stats(widths)
        ih_stats = size_stats(heights)

        return {
            "time": time_stats,
            "image_width": iw_stats,
            "image_height": ih_stats,
        }
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok"}
