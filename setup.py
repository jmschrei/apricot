from setuptools import setup

setup(
    name='apricot-select',
    version='0.2.2',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['apricot'],
    url='http://pypi.python.org/pypi/apricot-select/',
    license='LICENSE.txt',
    description='apricot is a package for submodular selection of representative sets for machine learning models.',
    install_requires=[
        "numpy >= 1.14.2",
        "scipy >= 1.0.0",
        "numba >= 0.39.0",
        "tqdm >= 4.24.0"
    ],
)
