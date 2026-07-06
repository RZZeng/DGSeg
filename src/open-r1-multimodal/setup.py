from setuptools import find_packages, setup


setup(
    name="open-r1-dgseg",
    version="0.1.0",
    description="GRPO training components used by DGSeg",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
)
