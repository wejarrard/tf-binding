from setuptools import setup, find_packages

setup(
    name="tf-binding-sites",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here from requirements.txt
        "torch>=2.1.0",
        "einops>=0.7.0",
        "polars>=0.19.12",
        "pysam>=0.22.0",
        "pyfaidx>=0.7.2.2",
        "tqdm>=4.66.1",
        "orjson"
    ],
)