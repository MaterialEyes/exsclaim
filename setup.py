#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
import glob
from shutil import rmtree

from setuptools import find_packages, setup, Command
from setuptools.command.install import install

# Package meta-data.
NAME = 'exsclaim'
DESCRIPTION = 'EXSCLAIM! is a library for the automatic EXtraction, Separation, and Caption-based natural Language Annotation of IMages from scientific figures.'
URL = 'https://github.com/MaterialEyes/exsclaim'
EMAIL = 'developers@materialeyes.org'
AUTHOR = 'Eric Schwenker','Trevor Spreadbury','Weixin Jiang','Maria Chan'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'

# What packages are required for this module to be executed?
REQUIRED = [
        'beautifulsoup4>=4.8.1',
        'bleach>=2.1.0', 
        'lxml>=4.4.1',
        'opencv-python',
        'pillow',
        'pyyaml',
        'Pygments',
        'matplotlib',
        'selenium',
        'scipy',
        'scikit-image',
        'spacy>=2.0.0,<3.0.0',
        'torch>=1.3.0',
        'torchvision>=0.4'
    ],

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

listOfFiles = list()
dirName = "exsclaim/journals/Google Chrome Canary.app/"
for (dirpath, dirnames, filenames) in os.walk(dirName):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()

# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(include=['exsclaim', 'exsclaim.*']),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    data_files=[(sys.prefix+'/captions/models/reference.yml',['exsclaim/captions/models/reference.yml']),
                (sys.prefix+'/captions/models/patterns.yml',['exsclaim/captions/models/patterns.yml']),
                (sys.prefix+'/captions/models/rules.yml',['exsclaim/captions/models/rules.yml']),
                (sys.prefix+'/captions/models/characterization.yml',['exsclaim/captions/models/characterization.yml']),
                (sys.prefix+'/figures/config/yolov3_default_master.cfg',['exsclaim/figures/config/yolov3_default_master.cfg']),
                (sys.prefix+'/figures/config/yolov3_default_subfig.cfg',['exsclaim/figures/config/yolov3_default_subfig.cfg'])
    ],
    dependency_links=['https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz#egg=package-1.0'],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)