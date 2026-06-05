from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from ppocrv6_dxnn import PPOCRv6DetOnnxRecDxnn, PPOCRv6Dxnn, PPOCRv6OnnxStatic
from ppocrv6_dxnn.config import load_asset_config
from ppocrv6_dxnn.reporting import ensure_report_dir, timestamp, write_benchmark_markdown, write_json


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


def time_pipeline(ocr, image, warmup: int, loops: int) -> tuple[int, float, float]:
    for _ in range(warmup):
        ocr(image)
    samples = []
    line_count = 0
    for _ in range(loops):
        start = time.perf_counter()
        results = ocr(image)
        samples.append((time.perf_counter() - start) * 1000.0)
        line_count = len(results)
    return line_count, statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def fmt_stat(mean_ms: float, std_ms: float) -> str:
    return f"{mean_ms:.1f}ms ± {std_ms:.1f}ms"


def parse_args() -> argparse.Namespace:
    assets = load_asset_config()
    parser = argparse.ArgumentParser(description="Benchmark PP-OCRv6 DXNN/ONNX static pipelines.")
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Image file or directory. Defaults to assets/general_ocr_002.png plus test_images/*.png.",
    )
    parser.add_argument("--backend", choices=["dxnn", "hybrid", "onnx", "both"], default="both")
    parser.add_argument("--det-dxnn", type=Path, default=assets.det_dxnn)
    parser.add_argument("--rec-dxnn", type=Path, default=assets.rec_dxnn)
    parser.add_argument("--det-onnx", type=Path, default=assets.det_onnx)
    parser.add_argument("--rec-onnx", type=Path, default=assets.rec_onnx)
    parser.add_argument("--dict", type=Path, default=assets.char_dict)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown report output.",
    )
    parser.set_defaults(asset_config=assets)
    return parser.parse_args()


def make_runner(name: str, args: argparse.Namespace):
    if name == "dxnn":
        return PPOCRv6Dxnn(args.det_dxnn, args.rec_dxnn, args.dict)
    if name == "hybrid":
        return PPOCRv6DetOnnxRecDxnn(args.det_onnx, args.rec_dxnn, args.dict)
    return PPOCRv6OnnxStatic(args.det_onnx, args.rec_onnx, args.dict)


def main() -> None:
    args = parse_args()
    if args.warmup < 0:
        raise SystemExit(f"--warmup must be >= 0, got {args.warmup}")
    if args.loops < 1:
        raise SystemExit(f"--loops must be >= 1, got {args.loops}")
    eval_image_paths = args.asset_config.eval_image_paths
    images = iter_images(args.images) if args.images else collect_images(eval_image_paths)
    if not images:
        raise SystemExit(f"No images found: {args.images or eval_image_paths}")

    backends = ["dxnn", "onnx"] if args.backend == "both" else [args.backend]
    print("=" * 100)
    label = "DXNN vs Static ONNX" if args.backend == "both" else args.backend.upper()
    print(f" PP-OCRv6 Benchmark: {label}")
    print("=" * 100)
    print(f" Images: {len(images)}   Warmup: {args.warmup}   Iters: {args.loops}")
    print()
    report = {
        "type": "benchmark",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_count": len(images),
        "images_path": str(args.images) if args.images else [str(path) for path in eval_image_paths],
        "warmup": args.warmup,
        "loops": args.loops,
        "backends": [],
        "comparison": {},
    }
    runtime_errors = 0
    for backend in backends:
        print(f"== {backend.upper()} ==")
        print(f" {'Image':<28s} {'#':>3s}  {'Latency':<22s}")
        print("-" * 60)
        backend_report = {
            "name": backend,
            "images": [],
            "summary": {},
            "error": None,
        }
        try:
            with make_runner(backend, args) as ocr:
                latencies = []
                for image_path in images:
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image is None:
                        print(f"skip unreadable image: {image_path}", file=sys.stderr)
                        continue
                    lines, mean_ms, std_ms = time_pipeline(ocr, image, args.warmup, args.loops)
                    latencies.append(mean_ms)
                    backend_report["images"].append({
                        "image": image_path.name,
                        "path": str(image_path),
                        "lines": lines,
                        "mean_ms": mean_ms,
                        "std_ms": std_ms,
                    })
                    print(f" {image_path.name:<28s} {lines:>3d}  {fmt_stat(mean_ms, std_ms):<22s}")
                if latencies:
                    backend_report["summary"] = {
                        "avg_mean_ms": statistics.mean(latencies),
                        "min_mean_ms": min(latencies),
                        "max_mean_ms": max(latencies),
                        "total_lines": sum(item["lines"] for item in backend_report["images"]),
                        "image_count": len(backend_report["images"]),
                    }
                    print("-" * 60)
                    print(
                        f"  {backend.upper():<10s} avg: {backend_report['summary']['avg_mean_ms']:.1f}ms "
                        f"(range: {backend_report['summary']['min_mean_ms']:.1f} ~ "
                        f"{backend_report['summary']['max_mean_ms']:.1f})"
                    )
        except Exception as exc:
            if backend in {"dxnn", "hybrid"}:
                print(f"{backend.upper()} benchmark failed: {exc}", file=sys.stderr)
                print("Check DX_RT_PATH, dxrt service, driver, and NPU availability.", file=sys.stderr)
                backend_report["error"] = str(exc)
                runtime_errors += 1
            else:
                raise
        report["backends"].append(backend_report)
        print()

    by_name = {backend["name"]: backend for backend in report["backends"]}
    dxnn_summary = by_name.get("dxnn", {}).get("summary") if by_name.get("dxnn") else None
    onnx_summary = by_name.get("onnx", {}).get("summary") if by_name.get("onnx") else None
    if dxnn_summary and onnx_summary:
        dxnn_avg = dxnn_summary["avg_mean_ms"]
        onnx_avg = onnx_summary["avg_mean_ms"]
        report["comparison"] = {
            "dxnn_avg_mean_ms": dxnn_avg,
            "onnx_avg_mean_ms": onnx_avg,
            "speedup_vs_onnx": onnx_avg / dxnn_avg if dxnn_avg > 0 else None,
        }
        print("=" * 100)
        print(
            f"  DXNN avg: {dxnn_avg:.1f}ms   ONNX avg: {onnx_avg:.1f}ms   "
            f"speedup: {report['comparison']['speedup_vs_onnx']:.2f}x"
        )
        print("=" * 100)

    report_dir = ensure_report_dir(args.report_dir or Path("reports") / f"benchmark_{timestamp()}")
    write_json(report_dir / "report.json", report)
    write_benchmark_markdown(report_dir / "report.md", report)
    print(f"\nReport saved to: {report_dir}")
    if runtime_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
