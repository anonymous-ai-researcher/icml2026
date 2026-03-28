from setuptools import setup, find_packages

setup(
    name="eeag-learning",
    version="1.0.0",
    description="Sample Complexity of Envy Elimination under Unknown Submodular Valuations",
    author="Anonymous",
    author_email="anonymous@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black", "flake8"],
    },
)
