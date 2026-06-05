from __future__ import annotations

import unittest

import cv2

from ppocrv6_dxnn import PPOCRv6OnnxStatic
from ppocrv6_dxnn.config import load_asset_config


class StaticOnnxPipelineTest(unittest.TestCase):
    def test_static_onnx_pipeline_runs_demo_image(self) -> None:
        assets = load_asset_config()
        image = cv2.imread(str(assets.demo_image), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        with PPOCRv6OnnxStatic(assets.det_onnx, assets.rec_onnx, assets.char_dict) as ocr:
            results = ocr(image)
        self.assertIsInstance(results, list)
        self.assertTrue(all(result.box and isinstance(result.text, str) for result in results))


if __name__ == "__main__":
    unittest.main()
