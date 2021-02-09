import setuptools
import pathlib

here = pathlib.Path(__file__).parent.resolve()

with open(here / "README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="exsclaim-materialeyes",
    version="0.0.15",
    author=('Eric Schwenker','Trevor Spreadbury','Weixin Jiang','Maria Chan'),
    author_email="developer@materialeyes.org",
    description="EXSCLAIM! is a library for the automatic EXtraction, Separation, and Caption-based natural Language Annotation of IMages from scientific figures.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MaterialEyes/exsclaim",
    packages=setuptools.find_packages(),
    install_requires= [
        "numpy",
        "pillow",
        "beautifulsoup4",
        "pyyaml>=5.1",
        "scipy",
        "scikit-image",
        "torch>=1.6",
        "opencv-python",
        "requests",
        "torchvision",
        "urllib3",
        "spacy",
        "lxml"
    ],
    package_data={
        'exsclaim': ['figures/config/yolov3_default_master.cfg',
                     'figures/config/yolov3_default_subfig.cfg',
                     'captions/models/characterization.yml',
                     'captions/models/patterns.yml',
                     'captions/models/reference.yml',
                     'captions/models/rules.yml',
                     'tests/data/nature_test.json']
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    project_urls={
        'Documentation': 'https://github.com/MaterialEyes/exsclaim/wiki',
        'Source': 'https://github.com/MaterialEyes/exsclaim',
        'Tracker': 'https://github.com/MaterialEyes/exsclaim/issues',
        },
)