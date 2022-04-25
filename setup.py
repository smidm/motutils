#!/usr/bin/env python
# coding=utf-8
import io
import re
from os.path import dirname, join

from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        if not kwargs.get("requirements", False):
            return fh.read()
        else:
            out = []
            for line in fh.readlines():
                out.append(re.sub(r"(git\+.*egg=(.*))", "\2 @ \1", line))
            return out


setup(
    name="motutils",
    version="0.1.1",
    license="MIT",
    description="utilities for multiple object tracking research (io, visualization, format conversion)",
    long_description=re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
        "", read("README.md")
    ),
    long_description_content_type="text/markdown",
    author="Matěj Šmíd",
    author_email="m@matejsmid.cz",
    url="https://github.com/smidm/motutils",
    packages=["motutils"],
    install_requires=[
        "click",
        "h5py",
        "matplotlib",
        "motmetrics",
        "moviepy",
        "numpy",
        "opencv-python-headless",
        "pandas",
        "scipy",
        "tqdm",
        "xarray",
        "shape",
    ],
    # read('requirements.txt', requirements=True),
    extras_require={"sleap": "sleap"},
    python_requires=">=3.6",
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        # 'Changelog': 'https://github.com/smidm/motutils/blob/master/CHANGELOG.md',
        "Issue Tracker": "https://github.com/smidm/motutils/issues",
    },
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "motutils=motutils.cli:cli",
        ],
    },
)
