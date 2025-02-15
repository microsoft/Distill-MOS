from setuptools import setup, find_packages

setup(
    name="distillmos",
    version="0.9.1",
    description="Efficient speech quality assessment learned from SSL-based speech quality assessment model",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Microsoft | Benjamin Stahl, Hannes Gamper",
    packages=find_packages(),
    license="MIT",
    include_package_data=True,
    package_data={"distillmos": ["weights/*"]},
    python_requires=">=3.8",
    install_requires=[
        "xls-r-sqa==0.1.0",
        "torch>=1.11.0",
        "numpy>=1.23.5",
        "soundfile",
        "torchaudio",
    ],
    extras_require={
        "dev": ["pytest", "requests"]
    },
    entry_points={
        "console_scripts": [
            "distillmos = distillmos.sqa:command_line_inference",
        ],
    },
)
