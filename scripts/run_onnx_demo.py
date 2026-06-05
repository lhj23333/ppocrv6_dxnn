from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from ppocrv6_dxnn import PPOCRv6OnnxStatic
from ppocrv6_dxnn.config import load_asset_config


def parse_args() -> argparse.Namespace:
    assets = load_asset_config()
    parser = argparse.ArgumentParser(description="Run the static ONNX reference pipeline.")
    parser.add_argument("image", nargs="?", type=Path, default=assets.demo_image)
    parser.add_argument("--det-onnx", type=Path, default=assets.det_onnx)
    parser.add_argument("--rec-onnx", type=Path, default=assets.rec_onnx)
    parser.add_argument("--dict", type=Path, default=assets.char_dict)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    with PPOCRv6OnnxStatic(args.det_onnx, args.rec_onnx, args.dict) as ocr:
        results = ocr(image)
    print(f"Detected {len(results)} text regions")
    for idx, result in enumerate(results, 1):
        print(f"[{idx:02d}] score={result.score:.6f} text={result.text}")


if __name__ == "__main__":
    main()
