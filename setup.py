from setuptools import setup, find_packages

setup(
    name='Coordinate Backoff Bayesian optimization',
    version='1.0.0',
    url='CobBO',
    packages=find_packages(),
    author='Jian Tan, Niv Nayman',
    author_email="j.tan@alibaba-inc.com, niv.nayman@alibaba-inc.com",
    description='A python implementation of CobBO',
    long_description='A Python implementation of Coordinate Backoff Bayesian Optimization.',
    download_url='CobBO',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.18.0",
    ],
)