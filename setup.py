import os
from setuptools import find_namespace_packages, setup

setup(
    name="spatialforcing-vln",
    version="0.1.0",
    packages=find_namespace_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "accelerate==1.13.0",
        "datasets==3.6.0",
        "decord==0.6.0",
        "deepspeed==0.16.4",
        "einops>=0.8.0",
        "huggingface-hub>=0.34.0",
        "numpy>=1.26,<3",
        "packaging>=24.0",
        "pillow",
        "safetensors>=0.5.0",
        "transformers==5.3.0",
    ],
    extras_require={"dev": ["pytest>=9.0"]},
    author="SpatialForcing-VLN contributors",
    description="Training-only spatial representation alignment for Qwen3.5 VLN",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12,<3.13",
)
