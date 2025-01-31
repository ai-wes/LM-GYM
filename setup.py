from setuptools import setup, find_packages

setup(
    name="llm_gym",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "pyyaml>=5.4.0",
    ],
    author="LLM GYM Team",
    description="A framework for training and evaluating LLM cognitive strategies through reinforcement learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
) 