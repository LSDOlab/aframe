from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='aframe',
    version='0.1.0',
    author='Nicholas Orndorff',
    author_email='norndorff@ucsd.edu',
    license='MIT',
    url='https://github.com/LSDOlab/aframe',
    download_url='https://github.com/LSDOlab/aframe',
    description='A linear beam solver written in CSDL',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.7',
    platforms=['any'],
    install_requires=[
        'numpy',
        'pytest',
        'myst-nb',
        'sphinx',
        'sphinx_rtd_theme',
        'sphinx-copybutton',
        'sphinx-autoapi',
        'numpydoc',
        'gitpython',
        'sphinxcontrib-bibtex',
        'setuptools',
        'wheel',
        'twine'
    ],
)
