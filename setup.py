from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')
setup(
    name='copygec',
    version='0.0.1',
    packages=find_packages(where='copygec'),
)