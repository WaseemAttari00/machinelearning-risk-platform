"""
Package setup file.

Why have a setup.py?
  Without it, Python can only import modules from the current working directory.
  With `pip install -e .` (editable install), Python adds this project to sys.path,
  allowing `from src.models.train import ...` to work from any directory.

  This is important for:
  - Running tests from any directory: `pytest tests/`
  - Docker: the PYTHONPATH env var achieves the same result in containers
  - Clean imports: no sys.path manipulation needed in scripts
"""

from setuptools import setup, find_packages

setup(
    name="intelligent-risk-analytics",
    version="1.0.0",
    description="End-to-end ML platform for risk prediction (credit + network intrusion)",
    author="Waseem Attari",
    packages=find_packages(include=["src", "src.*", "api", "api.*"]),
    python_requires=">=3.11",
    install_requires=[],   # Dependencies managed via requirements.txt
)
