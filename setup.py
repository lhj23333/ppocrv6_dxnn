from setuptools import setup


setup(
    name="ppocrv6-dxnn",
    version="0.1.0",
    description="PP-OCRv6 DXNN inference pipeline with static ONNX reference checks.",
    packages=["ppocrv6_dxnn"],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "opencv-python",
        "onnxruntime",
        "pyclipper",
    ],
)
