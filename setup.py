from setuptools import setup, find_packages

setup(
    name="intelligent-risk-analytics",
    version="1.0.0",
    description="End-to-end ML platform for risk prediction (credit + network intrusion)",
    author="Waseem Attari",
    packages=find_packages(include=["src", "src.*", "api", "api.*"]),
    python_requires=">=3.11",
    install_requires=[],
)
