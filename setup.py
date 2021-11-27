from pathlib import Path
from setuptools import setup, find_packages

base_dir = Path(__file__).resolve().parent
    
setup(
    name = "copygec",
    version = "0.0.1",
    license = "MIT",
    description = "copygec",
    author = "Aleksander Misztal",
    python_requires = ">= 3.3",
    packages = find_packages(),    
    include_package_data=True,
)