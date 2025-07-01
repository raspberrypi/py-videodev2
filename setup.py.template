from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Python ctypes bindings for Video4Linux2 (V4L2) API"

setup(
    name="videodev2",
    version="0.0.1",
    description="Python ctypes bindings for Video4Linux2 (V4L2) API",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Raspberry Pi",
    license = "BSD-3-Clause",
    url="https://github.com/raspberrypi/py-videodev2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    install_requires=[],
    extras_require={
        "dev": ["twine", "wheel", "build"],
    },
    keywords="v4l2 video4linux2 linux video capture ctypes",
    project_urls={
        "Bug Reports": "https://github.com/raspberrypi/py-videodev2/issues",
        "Source": "https://github.com/raspberrypi/py-videodev2",
    },
)
