from setuptools import setup, find_packages

_wiser_requires = [
    "mani_skill>=3.0.0",
    "lerobot>=0.4.3",
    "tensordict",
]

_gwm_wiser_requires = _wiser_requires + [
    "transformers==4.57.6",
    "scikit-learn",
]

extras_require = {
    "wiser": _wiser_requires,
    "gwm-wiser": _gwm_wiser_requires,
}

setup(
    name="gwm_wiser",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "gwm_wiser": [
            "assets/configs/*.yaml",
            "assets/images/*/*.png",
            "assets/images/*/*.jpg",
            "assets/images/*/*.json",
        ],
    },
    # No base dependencies – everything lives in extras
    install_requires=[],
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
