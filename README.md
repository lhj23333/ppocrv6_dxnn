# PP-OCRv6 DXNN Inference

PP-OCRv6 DXNN inference, benchmark, and precision validation against static
ONNX reference models.

This repo is self-contained: DXNN/ONNX models, dictionary, demo image, and test
images are stored locally. Runtime asset paths are configured in
`asset_config.json`.

## Quick Start

### 1. Install

```bash
cd ppocrv6_dxnn

./install.sh
source .venv/bin/activate
```

Requires Python `3.10+`.

### 2. Configure DX Runtime

DXNN commands require the DEEPX DX Runtime SDK:

```bash
export DX_RT_PATH=/path/to/dx-all-suite/dx-runtime/dx_rt
```

### 3. Run Demo

```bash
python demo.py
```

The ONNX reference demo does not require DX Runtime:

```bash
python scripts/run_onnx_demo.py
```

## Benchmark

Compare DXNN with the static ONNX reference:

```bash
python benchmark.py
```

Run one backend only:

```bash
python benchmark.py --backend dxnn
python benchmark.py --backend onnx
python benchmark.py --backend hybrid
```

Default benchmark settings match the ONNX repo: demo image + `test_images/*.png`,
3 warmup runs, and 10 measured iterations per image.

## Precision Validation

Single-image validation:

```bash
python scripts/verify_dxnn_vs_onnx.py --mode single
```

Batch validation:

```bash
python scripts/verify_dxnn_vs_onnx.py --mode batch
```

Hybrid validation, using detection ONNX + recognition DXNN:

```bash
python scripts/verify_dxnn_vs_onnx.py --mode batch --candidate hybrid
```

Validation is tolerance-based because DXNN uses compiled/quantized runtime
models while the reference backend uses float static ONNX models.

## Tests

```bash
python -m unittest discover -s tests
```

## Assets

```text
asset_config.json
models/
  ppocrv6_det.dxnn
  ppocrv6_det.onnx
  ppocrv6_rec.dxnn
  ppocrv6_rec.onnx
  rec_char_dict.txt
assets/
  general_ocr_002.png
test_images/
  *.png
```

Edit `asset_config.json` to replace default models or test assets. Command-line
flags such as `--det-dxnn`, `--rec-dxnn`, `--det-onnx`, `--rec-onnx`, `--dict`,
and `--images` can override the JSON defaults for a single run.

## Project Structure

```text
ppocrv6_dxnn/
  ppocrv6_dxnn/              # OCR pipeline and runtime adapters
  demo.py                    # DXNN demo
  benchmark.py               # DXNN/ONNX benchmark
  scripts/
    run_onnx_demo.py         # Static ONNX reference demo
    verify_dxnn_vs_onnx.py   # Precision validation
  tests/                     # ONNX smoke test
```
