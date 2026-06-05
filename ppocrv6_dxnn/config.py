from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_CONFIG = REPO_ROOT / "asset_config.json"


@dataclass(frozen=True)
class AssetConfig:
    det_dxnn: Path
    rec_dxnn: Path
    det_onnx: Path
    rec_onnx: Path
    char_dict: Path
    demo_image: Path
    test_images: Path
    eval_image_paths: tuple[Path, ...]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"asset config section must be an object: {name}")
    return section


def _string(section: dict[str, Any], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"asset config value must be a non-empty string: {name}")
    return value


def _strings(section: dict[str, Any], name: str) -> tuple[str, ...]:
    value = section.get(name)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"asset config value must be a string list: {name}")
    return tuple(value)


@lru_cache(maxsize=1)
def load_asset_config(config_path: str | Path = DEFAULT_ASSET_CONFIG) -> AssetConfig:
    path = Path(config_path)
    root = path.resolve().parent
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"asset config root must be an object: {path}")

    models = _section(data, "models")
    data_section = _section(data, "data")

    return AssetConfig(
        det_dxnn=_resolve(root, _string(models, "det_dxnn")),
        rec_dxnn=_resolve(root, _string(models, "rec_dxnn")),
        det_onnx=_resolve(root, _string(models, "det_onnx")),
        rec_onnx=_resolve(root, _string(models, "rec_onnx")),
        char_dict=_resolve(root, _string(data_section, "char_dict")),
        demo_image=_resolve(root, _string(data_section, "demo_image")),
        test_images=_resolve(root, _string(data_section, "test_images")),
        eval_image_paths=tuple(
            _resolve(root, item) for item in _strings(data_section, "eval_image_paths")
        ),
    )
