"""
Setup script for AI-Generated Voice Detection API
"""

from setuptools import setup, find_packages

setup(
    name="ai-voice-detection-api",
    version="1.0.0",
    description="AI-Generated Voice Detection API",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "numpy>=1.24.3",
        "librosa>=0.10.1",
        "tensorflow>=2.15.0",
        "python-multipart>=0.0.6"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "hypothesis>=6.92.1",
            "httpx>=0.25.2"
        ]
    }
)