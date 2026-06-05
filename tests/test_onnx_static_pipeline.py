from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ppocrv6_dxnn import PPOCRv6OnnxStatic
from ppocrv6_dxnn.config import load_asset_config
from ppocrv6_dxnn.core import CTCLabelDecode, DBPostProcess
from scripts.verify_dxnn_vs_onnx import unreadable_image_stats


class StaticOnnxPipelineTest(unittest.TestCase):
    def test_static_onnx_pipeline_runs_demo_image(self) -> None:
        assets = load_asset_config()
        image = cv2.imread(str(assets.demo_image), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        with PPOCRv6OnnxStatic(assets.det_onnx, assets.rec_onnx, assets.char_dict) as ocr:
            results = ocr(image)
        self.assertIsInstance(results, list)
        self.assertTrue(all(result.box and isinstance(result.text, str) for result in results))

    def test_ctc_decode_accepts_2d_and_flat_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dict_path = Path(tmpdir) / "dict.txt"
            dict_path.write_text("a\nb\n", encoding="utf-8")
            decoder = CTCLabelDecode(dict_path)

            logits = np.zeros((3, decoder.vocab_size), dtype=np.float32)
            logits[0, 1] = 0.9
            logits[1, 1] = 0.8
            logits[2, 2] = 0.7

            texts_2d, _ = decoder(logits)
            texts_flat, _ = decoder(logits.reshape(-1))

        self.assertEqual(texts_2d, ["ab"])
        self.assertEqual(texts_flat, ["ab"])

    def test_ctc_decode_rejects_vocab_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dict_path = Path(tmpdir) / "dict.txt"
            dict_path.write_text("a\n", encoding="utf-8")
            decoder = CTCLabelDecode(dict_path)

            with self.assertRaisesRegex(ValueError, "vocab dimension mismatch"):
                decoder(np.zeros((1, 2, decoder.vocab_size + 1), dtype=np.float32))

    def test_db_postprocess_returns_int32_boxes(self) -> None:
        post = DBPostProcess(thresh=0.3, box_thresh=0.1, min_size=1)
        prob = np.ones((10, 10), dtype=np.float32)
        bitmap = prob > 0.3

        boxes, _ = post._extract_boxes(prob, bitmap, dst_w=40000, dst_h=40000)

        self.assertEqual(boxes.dtype, np.int32)

    def test_unreadable_image_stats_is_failure_shaped(self) -> None:
        stats = unreadable_image_stats("Cannot read image: broken.png")

        self.assertEqual(stats["dxnn_count"], 0)
        self.assertEqual(stats["onnx_count"], 0)
        self.assertEqual(stats["mismatches"][0]["type"], "image")


if __name__ == "__main__":
    unittest.main()
