from setuptools import setup

setup(
    name='apricot',
    version='0.1.0',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['apricot'],
    url='http://pypi.python.org/pypi/apricot/',
    license='LICENSE.txt',
    description='apricot is a package for submodular selection of representative sets.',
    install_requires=[
        "numpy >= 1.8.0",
    ],
)
