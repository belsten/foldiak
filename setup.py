import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="foldiak",
    version="0.0.1",
    author="Alexander Belsten",
    author_email="belsten@berkeley.edu",
    description="Pytorch infrastructure for Foldiak model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/belsten/foldiak",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)