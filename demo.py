from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ppocrv6_dxnn import PPOCRv6Dxnn
from ppocrv6_dxnn.config import load_asset_config
from ppocrv6_dxnn.reporting import (
    ensure_report_dir,
    results_to_dicts,
    timestamp,
    write_demo_markdown,
    write_json,
)


def draw_results(image, results):
    canvas = image.copy()
    for result in results:
        pts = cv2.convexHull(np.array(result.box, dtype="int32").reshape(-1, 1, 2))
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
        x, y = result.box[0]
        cv2.putText(
            canvas,
            result.text[:24],
            (int(x), max(0, int(y) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def parse_args() -> argparse.Namespace:
    assets = load_asset_config()
    parser = argparse.ArgumentParser(description="Run PP-OCRv6 DXNN demo.")
    parser.add_argument("image", nargs="?", type=Path, default=assets.demo_image)
    parser.add_argument("--det-dxnn", type=Path, default=assets.det_dxnn)
    parser.add_argument("--rec-dxnn", type=Path, default=assets.rec_dxnn)
    parser.add_argument("--dict", type=Path, default=assets.char_dict)
    parser.add_argument("--save-vis", type=Path, default=Path("output_vis_dxnn.png"))
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown report output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    try:
        with PPOCRv6Dxnn(args.det_dxnn, args.rec_dxnn, args.dict) as ocr:
            results = ocr(image)
    except Exception as exc:
        print(f"DXNN inference failed: {exc}", file=sys.stderr)
        print("Check DX_RT_PATH, dxrt service, driver, and NPU device availability.", file=sys.stderr)
        raise SystemExit(2) from exc

    print("=" * 80)
    print(" PP-OCRv6 DXNN Demo")
    print("=" * 80)
    print(f" Image: {args.image}")
    print(f" Detected {len(results)} text regions")
    print("-" * 80)
    for idx, result in enumerate(results, 1):
        print(f" [{idx:02d}] score={result.score:.6f}  text={result.text}")
    print("=" * 80)

    vis = draw_results(image, results)
    cv2.imwrite(str(args.save_vis), vis)
    print(f"Visualization saved to: {args.save_vis}")

    report_dir = ensure_report_dir(args.report_dir or Path("reports") / f"demo_{timestamp()}")
    report = {
        "type": "demo",
        "backend": "dxnn",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image": str(args.image),
        "det_model": str(args.det_dxnn),
        "rec_model": str(args.rec_dxnn),
        "char_dict": str(args.dict),
        "visualization": str(args.save_vis),
        "text_region_count": len(results),
        "results": results_to_dicts(results),
    }
    write_json(report_dir / "report.json", report)
    write_demo_markdown(report_dir / "report.md", report)
    print(f"Report saved to: {report_dir}")


if __name__ == "__main__":
    main()
