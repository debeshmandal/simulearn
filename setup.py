import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="simulearn",
    version="0.0.1",
    author="Debesh Mandal",
    description="Machine learning for simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/debeshmandal/simulearn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
