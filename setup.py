"""Compatibility entry point for older pip versions.

The canonical metadata remains in ``pyproject.toml``; this mirrors the small
set of fields needed by legacy editable installs.
"""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="skillforge",
    version="0.1.0",
    description="Fitness-driven lifecycle management for reusable agent skills",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(include=["skillforge*", "skill_generation*", "agent_system*", "verl*"]),
    package_data={"verl": ["version/*", "trainer/config/*.yaml"]},
    extras_require={
        "llm": ["openai>=1.30", "anthropic>=0.30"],
        "dev": ["pytest>=8"],
        "training": [
            "accelerate",
            "datasets",
            "dill",
            "gym>=0.26,<1",
            "gymnasium>=0.29,<1",
            "hydra-core",
            "numpy",
            "pandas",
            "pyarrow>=19.0.0",
            "Pillow",
            "PyYAML",
            "ray[default]>=2.41.0,<2.51.0",
            "tensordict>=0.8,<=0.10,!=0.9.0",
            "torch>=2.0",
            "torchvision",
            "transformers<=4.57.3",
        ],
        "gpu": ["vllm>=0.8.5,<=0.11.0"],
    },
)
