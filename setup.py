import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lde-tobikuhlmann",
    version="0.0.1",
    author="Tobias Kuhlmann",
    author_email="tobiasmartinkuhlmann@gmail.com",
    description="routines for lde density estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tobikuhlmann/lde",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
)