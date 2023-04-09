from setuptools import setup, find_packages
setup(
    name='har',
    description='package for experimentation on har algorithms',
    version='1.0',
    author='Amine AMMOR',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
