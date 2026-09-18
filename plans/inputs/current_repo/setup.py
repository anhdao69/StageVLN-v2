from setuptools import find_namespace_packages, setup


setup(
    name="qwen35-vln-sft",
    version="0.1.0",
    packages=find_namespace_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "accelerate==1.13.0",
        "deepspeed==0.16.4",
        "pillow",
        "safetensors>=0.5.0",
        "transformers==5.3.0",
    ],
    extras_require={"dev": ["pytest>=9.0"]},
    description="Plain Qwen3.5-4B SFT for JanusVLN R2R",
    python_requires=">=3.12,<3.13",
)
