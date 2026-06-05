from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from ppocrv6_dxnn import OCRResult, PPOCRv6DetOnnxRecDxnn, PPOCRv6Dxnn, PPOCRv6OnnxStatic
from ppocrv6_dxnn.config import load_asset_config
from ppocrv6_dxnn.reporting import (
    ensure_report_dir,
    results_to_dicts,
    timestamp,
    write_json,
    write_verify_markdown,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]


def collect_images(paths: tuple[Path, ...]) -> list[Path]:
    images: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        for image in iter_images(path):
            resolved = image.resolve()
            if resolved not in seen:
                seen.add(resolved)
                images.append(image)
    return images


def candidate_label(candidate: str) -> str:
    if candidate == "hybrid":
        return "HYBRID(det ONNX + rec DXNN)"
    return "DXNN(det DXNN + rec DXNN)"


def make_candidate(args: argparse.Namespace):
    if args.candidate == "hybrid":
        return PPOCRv6DetOnnxRecDxnn(args.det_onnx, args.rec_dxnn, args.dict)
    return PPOCRv6Dxnn(args.det_dxnn, args.rec_dxnn, args.dict)


def compare_results(
    candidate: list[OCRResult],
    onnx: list[OCRResult],
    *,
    include_box: bool,
) -> dict[str, Any]:
    paired_count = min(len(candidate), len(onnx))
    text_mismatch = 0
    mismatches: list[dict[str, Any]] = []
    score_diffs = []
    box_diffs = []

    for i in range(paired_count):
        if candidate[i].text != onnx[i].text:
            text_mismatch += 1
            mismatches.append({
                "index": i,
                "type": "text",
                "candidate": candidate[i].text,
                "onnx": onnx[i].text,
            })
        score_diffs.append(abs(candidate[i].score - onnx[i].score))
        if include_box:
            a = np.asarray(candidate[i].box, dtype=np.float32)
            b = np.asarray(onnx[i].box, dtype=np.float32)
            if a.shape == b.shape:
                box_diffs.append(float(np.abs(a - b).max()))

    if len(candidate) != len(onnx):
        mismatches.append({
            "index": paired_count,
            "type": "count",
            "candidate": len(candidate),
            "onnx": len(onnx),
        })

    return {
        "dxnn_count": len(candidate),
        "onnx_count": len(onnx),
        "paired_count": paired_count,
        "text_mismatch": text_mismatch,
        "count_delta": abs(len(candidate) - len(onnx)),
        "mean_score_diff": float(np.mean(score_diffs)) if score_diffs else None,
        "max_score_diff": float(np.max(score_diffs)) if score_diffs else None,
        "max_box_diff": float(np.max(box_diffs)) if box_diffs else None,
        "mismatches": mismatches,
    }


def unreadable_image_stats(message: str) -> dict[str, Any]:
    return {
        "dxnn_count": 0,
        "onnx_count": 0,
        "paired_count": 0,
        "text_mismatch": 0,
        "count_delta": 0,
        "mean_score_diff": None,
        "max_score_diff": None,
        "max_box_diff": None,
        "mismatches": [
            {
                "index": 0,
                "type": "image",
                "candidate": message,
                "onnx": "",
            }
        ],
    }


def _is_number(value: object) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool)


def _within_tol(value: object, tolerance: float) -> bool:
    return _is_number(value) and math.isfinite(float(value)) and float(value) <= tolerance


def _format_console_float(value: object, fmt: str) -> str:
    width = 0
    if fmt.startswith(">"):
        width_text = ""
        for char in fmt[1:]:
            if not char.isdigit():
                break
            width_text += char
        width = int(width_text or 0)
    if not isinstance(value, (float, int)):
        return f"{'n/a':>{width}s}" if width else "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return f"{'n/a':>{width}s}" if width else "n/a"
    return format(value, fmt)


def stats_pass(stats: dict[str, Any], args: argparse.Namespace, *, include_box: bool) -> bool:
    score_ok = stats["paired_count"] == 0 or _within_tol(stats["max_score_diff"], args.score_tol)
    box_ok = not include_box or stats["paired_count"] == 0 or _within_tol(stats["max_box_diff"], args.box_tol)
    return (
        stats["count_delta"] == 0
        and stats["text_mismatch"] == 0
        and score_ok
        and box_ok
    )


def new_report(
    args: argparse.Namespace,
    *,
    mode: str,
    image_paths: list[Path],
    include_box: bool,
) -> dict[str, Any]:
    return {
        "type": "verify_dxnn_vs_onnx",
        "mode": mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_backend": args.candidate,
        "candidate_label": candidate_label(args.candidate),
        "image_count": len(image_paths),
        "images_path": [str(path) for path in image_paths],
        "det_dxnn": str(args.det_dxnn),
        "rec_dxnn": str(args.rec_dxnn),
        "det_onnx": str(args.det_onnx),
        "rec_onnx": str(args.rec_onnx),
        "score_tol": args.score_tol,
        "box_tol": args.box_tol if include_box else None,
        "summary": {
            "passed": 0,
            "failed": 0,
            "total_dxnn_count": 0,
            "total_onnx_count": 0,
            "total_paired_count": 0,
            "total_count_delta": 0,
            "total_text_mismatch": 0,
            "mean_score_diff": None,
            "max_score_diff": None,
            "max_box_diff": None,
        },
        "images": [],
        "error": None,
    }


def add_image_stats(report: dict[str, Any], image_path: Path, stats: dict[str, Any], ok: bool) -> None:
    summary = report["summary"]
    summary["passed" if ok else "failed"] += 1
    summary["total_dxnn_count"] += stats["dxnn_count"]
    summary["total_onnx_count"] += stats["onnx_count"]
    summary["total_paired_count"] += stats["paired_count"]
    summary["total_count_delta"] += stats["count_delta"]
    summary["total_text_mismatch"] += stats["text_mismatch"]
    report["images"].append({
        "image": image_path.name,
        "path": str(image_path),
        "status": "PASS" if ok else "FAIL",
        **stats,
    })


def finalize_report(report: dict[str, Any], score_means: list[float]) -> None:
    summary = report["summary"]
    if score_means:
        summary["mean_score_diff"] = float(np.mean(score_means))
    for item in report["images"]:
        if _is_number(item.get("max_score_diff")) and math.isfinite(item["max_score_diff"]):
            current = summary["max_score_diff"]
            summary["max_score_diff"] = item["max_score_diff"] if current is None else max(current, item["max_score_diff"])
        if _is_number(item.get("max_box_diff")) and math.isfinite(item["max_box_diff"]):
            current = summary["max_box_diff"]
            summary["max_box_diff"] = item["max_box_diff"] if current is None else max(current, item["max_box_diff"])


def print_runtime_error_rows(report: dict[str, Any], image_paths: list[Path], error: Exception) -> None:
    report["error"] = str(error)
    report["summary"]["failed"] = len(image_paths)
    report["images"] = [
        {
            "image": image_path.name,
            "path": str(image_path),
            "status": "ERROR",
            "dxnn_count": 0,
            "onnx_count": 0,
            "paired_count": 0,
            "text_mismatch": 0,
            "count_delta": 0,
            "mean_score_diff": None,
            "max_score_diff": None,
            "max_box_diff": None,
            "mismatches": [
                {
                    "index": 0,
                    "type": "runtime",
                    "candidate": str(error),
                    "onnx": "",
                }
            ],
            "dxnn_results": [],
            "onnx_results": [],
        }
        for image_path in image_paths
    ]


def write_report(report: dict[str, Any], args: argparse.Namespace, mode: str) -> Path:
    report_dir = ensure_report_dir(args.report_dir or Path("reports") / f"verify_{mode}_{timestamp()}")
    write_json(report_dir / "report.json", report)
    write_verify_markdown(report_dir / "report.md", report)
    print(f"Report saved to: {report_dir}")
    return report_dir


def run_single(args: argparse.Namespace) -> int:
    image_path = args.image
    report = new_report(args, mode="single", image_paths=[image_path], include_box=True)
    print("=" * 70)
    print(" PP-OCRv6 Candidate vs Static ONNX Precision Verification")
    print("=" * 70)
    print(f" Candidate: {candidate_label(args.candidate)}")
    print(f" Image: {image_path}")
    print()
    sys.stdout.flush()
    try:
        with make_candidate(args) as candidate_ocr, PPOCRv6OnnxStatic(
            args.det_onnx, args.rec_onnx, args.dict
        ) as onnx_ocr:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            candidate_results = candidate_ocr(image)
            onnx_results = onnx_ocr(image)
            stats = compare_results(candidate_results, onnx_results, include_box=True)
            ok = stats_pass(stats, args, include_box=True)
            add_image_stats(report, image_path, stats, ok)
            report["images"][0]["dxnn_results"] = results_to_dicts(candidate_results)
            report["images"][0]["onnx_results"] = results_to_dicts(onnx_results)
    except Exception as exc:
        print(f"Candidate verification failed before comparison: {exc}", file=sys.stderr)
        print("Check DX_RT_PATH, dxrt service, driver, and NPU availability.", file=sys.stderr)
        print_runtime_error_rows(report, [image_path], exc)
        write_report(report, args, "single")
        return 2

    item = report["images"][0]
    finalize_report(report, [item["mean_score_diff"]] if _is_number(item["mean_score_diff"]) else [])
    print(f" Candidate text regions: {item['dxnn_count']}")
    print(f" ONNX text regions     : {item['onnx_count']}")
    print(f" Count diff            : {item['count_delta']}")
    print(f" Text mismatches       : {item['text_mismatch']}")
    print(f" Mean score diff       : {_format_console_float(item['mean_score_diff'], '.8f')}")
    print(f" Max score diff        : {_format_console_float(item['max_score_diff'], '.8f')}")
    print(f" Max box diff          : {_format_console_float(item['max_box_diff'], '.4f')} px")
    print("=" * 70)
    if item["status"] == "PASS":
        print(" Conclusion: PASS - candidate and static ONNX are aligned under the configured tolerances")
    else:
        print(" Conclusion: FAIL - differences exceeded the configured tolerances")
    print("=" * 70)
    write_report(report, args, "single")
    return 0 if item["status"] == "PASS" else 1


def run_batch(args: argparse.Namespace) -> int:
    eval_image_paths = args.asset_config.eval_image_paths
    image_paths = iter_images(args.images) if args.images else collect_images(eval_image_paths)
    if not image_paths:
        raise SystemExit(f"No images found: {args.images or eval_image_paths}")

    report = new_report(args, mode="batch", image_paths=image_paths, include_box=False)
    print("=" * 90)
    print(" PP-OCRv6 Batch Precision Verification: Candidate vs Static ONNX")
    print("=" * 90)
    print(f" Images: {len(image_paths)}")
    print(f" Candidate: {candidate_label(args.candidate)}")
    print()
    print(
        f" {'Image':<28s} {'Status':<6s} {'Cand/ONNX':>10s} "
        f"{'TextDiff':>8s} {'AvgScoreDiff':>13s} {'MaxScoreDiff':>13s}"
    )
    print("-" * 90)
    sys.stdout.flush()

    score_means: list[float] = []
    try:
        with make_candidate(args) as candidate_ocr, PPOCRv6OnnxStatic(
            args.det_onnx, args.rec_onnx, args.dict
        ) as onnx_ocr:
            for image_path in image_paths:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    message = f"Cannot read image: {image_path}"
                    print(message, file=sys.stderr)
                    stats = unreadable_image_stats(message)
                    add_image_stats(report, image_path, stats, ok=False)
                    report["images"][-1]["dxnn_results"] = []
                    report["images"][-1]["onnx_results"] = []
                    continue
                candidate_results = candidate_ocr(image)
                onnx_results = onnx_ocr(image)
                stats = compare_results(candidate_results, onnx_results, include_box=False)
                ok = stats_pass(stats, args, include_box=False)
                add_image_stats(report, image_path, stats, ok)
                report["images"][-1]["dxnn_results"] = results_to_dicts(candidate_results)
                report["images"][-1]["onnx_results"] = results_to_dicts(onnx_results)
                if _is_number(stats["mean_score_diff"]) and math.isfinite(stats["mean_score_diff"]):
                    score_means.append(stats["mean_score_diff"])
                print(
                    f" {image_path.name:<28s} {'PASS' if ok else 'FAIL':<6s} "
                    f"{stats['dxnn_count']:>4d}/{stats['onnx_count']:<5d} "
                    f"{stats['text_mismatch']:>8d} "
                    f"{_format_console_float(stats['mean_score_diff'], '>13.6g')} "
                    f"{_format_console_float(stats['max_score_diff'], '>13.6g')}"
                )
    except Exception as exc:
        print(f"Candidate verification failed before comparison: {exc}", file=sys.stderr)
        print("Check DX_RT_PATH, dxrt service, driver, and NPU availability.", file=sys.stderr)
        print_runtime_error_rows(report, image_paths, exc)
        for item in report["images"]:
            print(
                f" {item['image']:<28s} {'ERROR':<6s} "
                f"{item['dxnn_count']:>4d}/{item['onnx_count']:<5d} "
                f"{item['text_mismatch']:>8d} "
                f"{_format_console_float(item['mean_score_diff'], '>13.6g')} "
                f"{_format_console_float(item['max_score_diff'], '>13.6g')}"
            )
        print("=" * 90)
        print(f" Result: 0/{len(image_paths)} images aligned")
        print(" Conclusion: ERROR - candidate runtime failed before comparison")
        print("=" * 90)
        write_report(report, args, "batch")
        return 2

    finalize_report(report, score_means)
    passed = report["summary"]["passed"]
    failed = report["summary"]["failed"]
    print("=" * 90)
    print(f" Result: {passed}/{len(report['images'])} images aligned")
    print(f" Total text lines: Candidate={report['summary']['total_dxnn_count']}  ONNX={report['summary']['total_onnx_count']}")
    print(f" Text mismatches: {report['summary']['total_text_mismatch']}")
    print(f" Mean score diff: {_format_console_float(report['summary']['mean_score_diff'], '.6g')}")
    print(f" Max score diff : {_format_console_float(report['summary']['max_score_diff'], '.6g')}")
    if failed == 0:
        print(" Conclusion: PASS - candidate and static ONNX are aligned under the configured tolerances")
    else:
        print(" Conclusion: FAIL - differences exceeded the configured tolerances")
    print("=" * 90)
    write_report(report, args, "batch")
    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    assets = load_asset_config()
    parser = argparse.ArgumentParser(
        description="Verify PP-OCRv6 DXNN output against the matching static ONNX reference."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="batch",
        help="single mirrors the ONNX repo single-image verifier; batch mirrors its multi-image verifier.",
    )
    parser.add_argument("--image", type=Path, default=assets.demo_image)
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Batch image file or directory. Defaults to assets/general_ocr_002.png plus test_images/*.png.",
    )
    parser.add_argument("--det-dxnn", type=Path, default=assets.det_dxnn)
    parser.add_argument("--rec-dxnn", type=Path, default=assets.rec_dxnn)
    parser.add_argument("--det-onnx", type=Path, default=assets.det_onnx)
    parser.add_argument("--rec-onnx", type=Path, default=assets.rec_onnx)
    parser.add_argument(
        "--candidate",
        choices=["dxnn", "hybrid"],
        default="dxnn",
        help="Candidate pipeline: dxnn=det DXNN + rec DXNN; hybrid=det ONNX + rec DXNN.",
    )
    parser.add_argument("--dict", type=Path, default=assets.char_dict)
    parser.add_argument("--score-tol", type=float, default=0.05)
    parser.add_argument("--box-tol", type=float, default=3.0)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown report output.",
    )
    parser.set_defaults(asset_config=assets)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "single":
        raise SystemExit(run_single(args))
    raise SystemExit(run_batch(args))


if __name__ == "__main__":
    main()
