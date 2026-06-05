from __future__ import annotations

import copy
import functools
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper


logger = logging.getLogger(__name__)
DX_RT_PATH_ENV = "DX_RT_PATH"

@dataclass(frozen=True, slots=True)
class OCRResult:
    text: str
    score: float
    box: List[List[int]]


class ModelRunner(Protocol):
    def run(self, x: np.ndarray) -> np.ndarray: ...
    def close(self) -> None: ...


class OnnxRunner:
    def __init__(self, model_path: str | Path) -> None:
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def run(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.session.run(None, {self.input_name: x})[0])

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]


def _candidate_dx_engine_paths() -> List[Path]:
    raw = os.environ.get(DX_RT_PATH_ENV, "")
    paths: List[Path] = []
    seen: set[Path] = set()
    for item in raw.split(os.pathsep):
        if not item:
            continue
        base = Path(item).expanduser()
        for path in (base, base / "src", base / "python_package/src"):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
    return paths


def _load_dx_engine():
    last_error: ImportError | None = None
    for path in _candidate_dx_engine_paths():
        if (path / "dx_engine/__init__.py").is_file():
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)
            try:
                from dx_engine import InferenceEngine, InferenceOption
                return InferenceEngine, InferenceOption
            except ImportError as exc:
                last_error = exc
                continue
    message = (
        "dx_engine is required for DXNN inference. Set DX_RT_PATH to the DX Runtime "
        "dx_rt directory, for example: export DX_RT_PATH=/path/to/dx-runtime/dx_rt. "
        "DX_RT_PATH may also point directly to dx_rt/python_package/src."
    )
    raise ImportError(message) from last_error


def _dtype_from_info(info: dict[str, Any]) -> np.dtype:
    dtype = info.get("dtype", np.float32)
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, type):
        return np.dtype(dtype)
    text = str(dtype).lower()
    if "uint8" in text:
        return np.dtype(np.uint8)
    if "int8" in text:
        return np.dtype(np.int8)
    if "float16" in text:
        return np.dtype(np.float16)
    return np.dtype(np.float32)


class DxnnRunner:
    def __init__(self, model_path: str | Path, *, use_ort: bool = True) -> None:
        InferenceEngine, InferenceOption = _load_dx_engine()
        option = InferenceOption()
        option.use_ort = use_ort
        self.engine = InferenceEngine(str(model_path), option)
        input_info = self.engine.get_input_tensors_info()[0]
        self.input_dtype = _dtype_from_info(input_info)

    def run(self, x: np.ndarray) -> np.ndarray:
        feed = np.ascontiguousarray(x.astype(self.input_dtype, copy=False))
        outputs = self.engine.run(feed)
        return np.asarray(outputs[0])

    def close(self) -> None:
        dispose = getattr(self.engine, "dispose", None)
        if callable(dispose):
            dispose()
        self.engine = None  # type: ignore[assignment]


def _require_file(path: str | Path, name: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{name}: file not found: {path}")


def _order_minarea_box_points(contour: np.ndarray) -> Tuple[List[np.ndarray], float]:
    rrect = cv2.minAreaRect(contour)
    pts = sorted(cv2.boxPoints(rrect), key=lambda p: p[0])
    tl, bl = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    tr, br = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
    return [pts[tl], pts[tr], pts[br], pts[bl]], min(rrect[1])


class DetPreProcessFixed960:
    def __init__(self, size: int = 960) -> None:
        self.size = size
        self._scale = np.float32(1.0 / 255.0)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        src_h, src_w = img.shape[:2]
        resized = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        x = (resized.astype(np.float32, copy=False) * self._scale - self._mean) / self._std
        x = x.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32, copy=False)
        shape = np.array(
            [src_h, src_w, self.size / float(src_h), self.size / float(src_w)],
            dtype=np.float32,
        )
        return x, shape


class DBPostProcess:
    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.5,
        max_candidates: int = 1000,
        min_size: int = 3,
    ) -> None:
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.max_candidates = max_candidates
        self.min_size = min_size

    @staticmethod
    def _box_score(bitmap: np.ndarray, points: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        box = points.astype(np.float32, copy=True)
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return float(cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0])

    def _unclip(self, box: np.ndarray) -> np.ndarray:
        area = cv2.contourArea(box)
        length = cv2.arcLength(box, closed=True)
        if length <= 0:
            return box
        distance = area * self.unclip_ratio / length
        po = pyclipper.PyclipperOffset()
        po.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        paths = po.Execute(distance)
        if not paths:
            return box
        return np.asarray(paths[0])

    def _extract_boxes(
        self,
        prob: np.ndarray,
        bitmap: np.ndarray,
        dst_w: int,
        dst_h: int,
    ) -> Tuple[np.ndarray, List[float]]:
        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = outs[1] if len(outs) == 3 else outs[0]
        ws, hs = dst_w / bitmap.shape[1], dst_h / bitmap.shape[0]
        boxes: List[np.ndarray] = []
        scores: List[float] = []

        for contour in contours[:self.max_candidates]:
            pts, sside = _order_minarea_box_points(contour)
            if sside < self.min_size:
                continue
            pts_np = np.array(pts, dtype=np.float32)
            score = self._box_score(prob, pts_np.reshape(-1, 2))
            if score < self.box_thresh:
                continue
            expanded = self._unclip(pts_np).reshape(-1, 1, 2)
            expanded_pts, sside2 = _order_minarea_box_points(expanded)
            if sside2 < self.min_size + 2:
                continue
            box = np.array(expanded_pts, dtype=np.float32)
            box[:, 0] = np.clip(np.round(box[:, 0] * ws), 0, dst_w)
            box[:, 1] = np.clip(np.round(box[:, 1] * hs), 0, dst_h)
            boxes.append(box.astype(np.int32))
            scores.append(score)

        if not boxes:
            return np.empty((0, 4, 2), dtype=np.int32), []
        return np.stack(boxes, axis=0), scores

    def __call__(self, pred: np.ndarray, img_shape: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        pred = np.asarray(pred)
        if pred.ndim == 3:
            pred = pred[:, np.newaxis, :, :]
        if pred.ndim == 1:
            side = int(math.sqrt(pred.size))
            if side * side != pred.size:
                raise ValueError(f"Cannot infer square detection output from flat size {pred.size}")
            pred = pred.reshape(1, 1, side, side)
        prob_map = pred[0, 0, :, :]
        segmentation = prob_map > self.thresh
        return self._extract_boxes(prob_map, segmentation, int(img_shape[1]), int(img_shape[0]))


def sort_quad_boxes(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) <= 1:
        return boxes
    items = list(sorted(boxes, key=lambda b: (b[0][1], b[0][0])))
    for i in range(len(items) - 1):
        for j in range(i, -1, -1):
            if abs(items[j + 1][0][1] - items[j][0][1]) < 10 and (
                items[j + 1][0][0] < items[j][0][0]
            ):
                items[j], items[j + 1] = items[j + 1], items[j]
            else:
                break
    return np.array(items, dtype=boxes.dtype)


def _rotate_crop_image(img: np.ndarray, points: np.ndarray) -> Optional[np.ndarray]:
    pts = points.astype(np.float32)
    crop_w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    crop_h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if crop_w < 1 or crop_h < 1:
        return None
    dst_pts = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])
    try:
        matrix = cv2.getPerspectiveTransform(pts, dst_pts)
    except cv2.error:
        return None
    cropped = cv2.warpPerspective(
        img,
        matrix,
        (crop_w, crop_h),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if cropped.shape[0] / cropped.shape[1] >= 1.5:
        cropped = np.rot90(cropped)
    return cropped


def _minarea_rect_crop(img: np.ndarray, poly: np.ndarray) -> Optional[np.ndarray]:
    box, _ = _order_minarea_box_points(poly.astype(np.int32))
    return _rotate_crop_image(img, np.array(box))


def crop_by_polys(img: np.ndarray, dt_polys: np.ndarray) -> List[Optional[np.ndarray]]:
    crops: List[Optional[np.ndarray]] = []
    for poly in dt_polys:
        crop = _minarea_rect_crop(img, copy.deepcopy(poly))
        crops.append(crop if crop is not None and crop.size > 0 else None)
    return crops


class RecPreProcessFixed320:
    def __init__(self, image_shape: Tuple[int, int, int] = (3, 48, 320)) -> None:
        self.channels, self.height, self.width = image_shape

    def _resize_norm(self, img: np.ndarray) -> np.ndarray:
        src_h, src_w = img.shape[:2]
        actual_w = min(int(math.ceil(self.height * src_w / float(src_h))), self.width)
        resized = cv2.resize(img, (actual_w, self.height), interpolation=cv2.INTER_LINEAR)
        chw = resized.astype(np.float32, copy=False).transpose(2, 0, 1)
        chw *= 1.0 / 255.0
        chw = (chw - 0.5) / 0.5
        padded = np.zeros((self.channels, self.height, self.width), dtype=np.float32)
        padded[:, :, :actual_w] = chw
        return padded

    def __call__(self, imgs: List[np.ndarray]) -> np.ndarray:
        return np.stack([self._resize_norm(img) for img in imgs], axis=0).astype(np.float32, copy=False)


class CTCLabelDecode:
    def __init__(self, character_dict_path: str | Path) -> None:
        with open(character_dict_path, encoding="utf-8") as f:
            raw = tuple(line.rstrip("\n\r") for line in f)
        self._blank = 0
        self._chars: Tuple[str, ...] = ("blank", *raw)

    @property
    def vocab_size(self) -> int:
        return len(self._chars)

    def __call__(self, model_output: np.ndarray) -> Tuple[List[str], List[float]]:
        output = np.asarray(model_output)
        if output.ndim == 1:
            if output.size % self.vocab_size != 0:
                raise ValueError(
                    "Cannot infer recognition output shape from flat size "
                    f"{output.size} and vocab size {self.vocab_size}"
                )
            output = output.reshape(1, output.size // self.vocab_size, self.vocab_size)
        elif output.ndim == 2:
            output = output[np.newaxis, :, :]
        elif output.ndim != 3:
            raise ValueError(f"Recognition output must be 1D, 2D, or 3D, got {output.ndim}D")
        if output.shape[-1] != self.vocab_size:
            raise ValueError(
                "Recognition output vocab dimension mismatch: "
                f"got {output.shape[-1]}, expected {self.vocab_size}"
            )
        indices = output.argmax(axis=-1)
        probs = output.max(axis=-1)
        texts: List[str] = []
        scores: List[float] = []
        for b in range(len(indices)):
            seq = indices[b]
            keep = np.ones(len(seq), dtype=bool)
            keep[1:] = seq[1:] != seq[:-1]
            keep &= seq != self._blank
            texts.append("".join(self._chars[int(idx)] for idx in seq[keep]))
            scores.append(float(probs[b][keep].mean()) if keep.any() else 0.0)
        return texts, scores


class PPOCRv6Base:
    def __init__(
        self,
        det_runner: ModelRunner,
        rec_runner: ModelRunner,
        rec_char_dict_path: str | Path,
        *,
        det_thresh: float = 0.3,
        det_box_thresh: float = 0.6,
        det_unclip_ratio: float = 1.5,
        rec_batch_size: int = 1,
    ) -> None:
        _require_file(rec_char_dict_path, "rec_char_dict_path")
        if rec_batch_size != 1:
            raise ValueError("Static DXNN rec model is fixed to batch size 1; use rec_batch_size=1.")
        self._det_runner = det_runner
        self._rec_runner = rec_runner
        self._det_pre = DetPreProcessFixed960()
        self._det_post = DBPostProcess(det_thresh, det_box_thresh, det_unclip_ratio)
        self._rec_pre = RecPreProcessFixed320()
        self._rec_post = CTCLabelDecode(rec_char_dict_path)
        self._closed = False

    @staticmethod
    def _require_open(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self: "PPOCRv6Base", *args: Any, **kwargs: Any) -> Any:
            if self._closed:
                raise RuntimeError("OCR runner is closed.")
            return method(self, *args, **kwargs)
        return wrapper

    def close(self) -> None:
        if self._closed:
            return
        self._det_runner.close()
        self._rec_runner.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @_require_open
    def detect(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        x, shape = self._det_pre(img_bgr)
        pred = self._det_runner.run(x)
        return self._det_post(pred, shape)

    @_require_open
    def recognize(self, img_list: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        texts: List[str] = []
        scores: List[float] = []
        for img in img_list:
            x = self._rec_pre([img])
            out = self._rec_runner.run(x)
            batch_texts, batch_scores = self._rec_post(out)
            texts.extend(batch_texts)
            scores.extend(batch_scores)
        return texts, scores

    @_require_open
    def __call__(self, img_bgr: np.ndarray) -> List[OCRResult]:
        boxes, _ = self.detect(img_bgr)
        if len(boxes) == 0:
            return []
        sorted_boxes = sort_quad_boxes(boxes)
        crops = crop_by_polys(img_bgr, sorted_boxes)
        valid_boxes: List[np.ndarray] = []
        valid_crops: List[np.ndarray] = []
        for box, crop in zip(sorted_boxes, crops):
            if crop is not None:
                valid_boxes.append(box)
                valid_crops.append(crop)
        if not valid_crops:
            return []
        texts, scores = self.recognize(valid_crops)
        return [
            OCRResult(text=text, score=score, box=box.tolist())
            for text, score, box in zip(texts, scores, valid_boxes)
        ]


class PPOCRv6Dxnn(PPOCRv6Base):
    def __init__(
        self,
        det_model_path: str | Path,
        rec_model_path: str | Path,
        rec_char_dict_path: str | Path,
        **kwargs: Any,
    ) -> None:
        _require_file(det_model_path, "det_model_path")
        _require_file(rec_model_path, "rec_model_path")
        super().__init__(
            DxnnRunner(det_model_path, use_ort=True),
            DxnnRunner(rec_model_path, use_ort=True),
            rec_char_dict_path,
            **kwargs,
        )


class PPOCRv6OnnxStatic(PPOCRv6Base):
    def __init__(
        self,
        det_model_path: str | Path,
        rec_model_path: str | Path,
        rec_char_dict_path: str | Path,
        **kwargs: Any,
    ) -> None:
        _require_file(det_model_path, "det_model_path")
        _require_file(rec_model_path, "rec_model_path")
        super().__init__(
            OnnxRunner(det_model_path),
            OnnxRunner(rec_model_path),
            rec_char_dict_path,
            **kwargs,
        )


class PPOCRv6DetOnnxRecDxnn(PPOCRv6Base):
    def __init__(
        self,
        det_model_path: str | Path,
        rec_model_path: str | Path,
        rec_char_dict_path: str | Path,
        **kwargs: Any,
    ) -> None:
        _require_file(det_model_path, "det_model_path")
        _require_file(rec_model_path, "rec_model_path")
        super().__init__(
            OnnxRunner(det_model_path),
            DxnnRunner(rec_model_path, use_ort=True),
            rec_char_dict_path,
            **kwargs,
        )
