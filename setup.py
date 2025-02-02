from setuptools import setup, find_packages

setup(
    name="distillmos",
    version="0.9.0",
    description="Efficient speech quality assessment learned from SSL-based speech quality assessment model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Microsoft | Benjamin Stahl, Hannes Gamper",
    packages=find_packages(),
    license="MIT",
    include_package_data=True,
    package_data={"distillmos": ["weights/*"]},
    python_requires=">=3.8",
    install_requires=[
        "xls_r_sqa @ git+https://github.com/lcn-kul/xls-r-analysis-sqa@fac0189e13d4be70b10e5a679bc966119d8b5432",
        "torch>=1.11.0",
        "numpy>=1.23.5",
        "soundfile",
        "torchaudio",
    ],
    entry_points={
        "console_scripts": [
            "distillmos = distillmos.sqa:command_line_inference",
        ],
    },
)