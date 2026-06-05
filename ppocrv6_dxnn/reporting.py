from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .core import OCRResult


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_report_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(data), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _format_float(value: Any, fmt: str) -> str:
    if not isinstance(value, (float, int)):
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    return format(value, fmt)


def result_to_dict(result: OCRResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "score": float(result.score),
        "box": result.box,
    }


def results_to_dicts(results: Iterable[OCRResult]) -> list[dict[str, Any]]:
    return [result_to_dict(result) for result in results]


def write_demo_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# PP-OCRv6 DXNN Demo Report",
        "",
        f"- Backend: `{report['backend']}`",
        f"- Image: `{report['image']}`",
        f"- Text regions: `{report['text_region_count']}`",
        f"- Visualization: `{report.get('visualization', '')}`",
        "",
        "| # | Score | Text |",
        "|---:|---:|---|",
    ]
    for idx, result in enumerate(report["results"], 1):
        text = str(result["text"]).replace("|", "\\|")
        lines.append(f"| {idx} | {result['score']:.6f} | {text} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# PP-OCRv6 Benchmark Report",
        "",
        f"- Images: `{report['image_count']}`",
        f"- Warmup: `{report['warmup']}`",
        f"- Loops: `{report['loops']}`",
        "",
    ]
    for backend in report["backends"]:
        lines.extend([
            f"## {backend['name'].upper()}",
            "",
        ])
        if backend.get("error"):
            lines.extend([f"Error: `{backend['error']}`", ""])
            continue
        lines.extend([
            f"- Average latency: `{backend['summary']['avg_mean_ms']:.3f} ms`",
            f"- Min latency: `{backend['summary']['min_mean_ms']:.3f} ms`",
            f"- Max latency: `{backend['summary']['max_mean_ms']:.3f} ms`",
            f"- Total lines: `{backend['summary'].get('total_lines', 0)}`",
            "",
            "| Image | Lines | Mean ms | Std ms |",
            "|---|---:|---:|---:|",
        ])
        for item in backend["images"]:
            lines.append(
                f"| {item['image']} | {item['lines']} | "
                f"{item['mean_ms']:.3f} | {item['std_ms']:.3f} |"
            )
        lines.append("")
    if report.get("comparison"):
        comparison = report["comparison"]
        lines.extend([
            "## Backend Comparison",
            "",
            f"- DXNN average: `{comparison['dxnn_avg_mean_ms']:.3f} ms`",
            f"- ONNX average: `{comparison['onnx_avg_mean_ms']:.3f} ms`",
            f"- Speedup vs ONNX: `{comparison['speedup_vs_onnx']:.3f}x`",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_verify_markdown(path: Path, report: dict[str, Any]) -> None:
    candidate_label = report.get("candidate_label", "DXNN")
    lines = [
        "# PP-OCRv6 Candidate vs ONNX Verification Report",
        "",
        f"- Candidate: `{candidate_label}`",
        f"- Images: `{report['image_count']}`",
        f"- Passed: `{report['summary']['passed']}`",
        f"- Failed: `{report['summary']['failed']}`",
        f"- Total lines Candidate/ONNX: `{report['summary'].get('total_dxnn_count', 0)}` / `{report['summary'].get('total_onnx_count', 0)}`",
        f"- Comparable pairs: `{report['summary'].get('total_paired_count', 0)}`",
        f"- Count diff total: `{report['summary'].get('total_count_delta', 0)}`",
        f"- Text mismatches: `{report['summary'].get('total_text_mismatch', 0)}`",
        f"- Mean score diff: `{_format_float(report['summary'].get('mean_score_diff'), '.6g')}`",
        f"- Max score diff: `{_format_float(report['summary'].get('max_score_diff'), '.6g')}`",
        f"- Score tolerance: `{report['score_tol']}`",
        "",
    ]
    if report.get("box_tol") is not None:
        lines.insert(-1, f"- Max box diff: `{_format_float(report['summary'].get('max_box_diff'), '.3f')}`")
        lines.insert(-1, f"- Box tolerance: `{report['box_tol']}`")
    if report.get("error"):
        lines.extend([f"Error: `{report['error']}`", ""])
    lines.extend([
        "| Image | Status | Count Candidate/ONNX | Count diff | Text mismatch | Max score diff | Max box diff |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for item in report["images"]:
        lines.append(
            f"| {item['image']} | {item['status']} | "
            f"{item['dxnn_count']}/{item['onnx_count']} | "
            f"{item.get('count_delta', 0)} | "
            f"{item['text_mismatch']} | {_format_float(item['max_score_diff'], '.6g')} | "
            f"{_format_float(item['max_box_diff'], '.3f')} |"
        )
    mismatch_lines = []
    for item in report["images"]:
        for mismatch in item.get("mismatches", []):
            mismatch_lines.append((item["image"], mismatch))
    if mismatch_lines:
        lines.extend([
            "",
            "## Mismatches",
            "",
            "| Image | Index | Type | Candidate | ONNX |",
            "|---|---:|---|---|---|",
        ])
        for image_name, mismatch in mismatch_lines:
            dxnn = str(mismatch.get("candidate", mismatch.get("dxnn", ""))).replace("|", "\\|")
            onnx = str(mismatch.get("onnx", "")).replace("|", "\\|")
            lines.append(
                f"| {image_name} | {mismatch.get('index', '')} | "
                f"{mismatch.get('type', '')} | {dxnn} | {onnx} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
