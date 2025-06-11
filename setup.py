from setuptools import setup, find_packages

setup(
    name="galCIB",
    version="0.1.0",
    description="A Python Package to calculate 3D and 2D galaxy-CIB clustering",
    author="Tanveer Karim",
    author_email="tanveer.karim@utoronto.ca",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS",
    ],
    install_requires=[
        "numpy>=1.24.4",
        "scipy>=1.14.0",
        "astropy>=5.1.1",
        "colossus>=1.3.8",
        "camb>=1.5.4"
    ],
    # extra_requires={
    #   "cosmology": [
    #       "camb>=1.5.4",
    #   ]  
    # },
    python_requires=">=3.10",
)