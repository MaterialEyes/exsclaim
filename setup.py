import setuptools
import pathlib

## build with  python setup.py bdist_wheel
## upload to testpypi w/  python3 -m twine upload --repository testpypi dist/*

here = pathlib.Path(__file__).parent.resolve()

with open(here / "README.md", "r") as fh:
    long_description = fh.read()

with open(here / "requirements.txt", "r") as f:
    install_requires = list(f.read().splitlines())

setuptools.setup(
    name="exsclaim-tspread",
    version="0.0.18",
    author=('Eric Schwenker','Trevor Spreadbury','Weixin Jiang','Maria Chan'),
    author_email="developer@materialeyes.org",
    description="EXSCLAIM! is a library for the automatic EXtraction, Separation, and Caption-based natural Language Annotation of IMages from scientific figures.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MaterialEyes/exsclaim",
    packages=setuptools.find_packages(),
    install_requires= install_requires,
    package_data={
        'exsclaim': ['figures/config/yolov3_default_master.cfg',
                     'figures/config/yolov3_default_subfig.cfg',
                     'figures/config/scale_label_reader.json'
                     'captions/models/characterization.yml',
                     'captions/models/patterns.yml',
                     'captions/models/reference.yml',
                     'captions/models/rules.yml',
                     'tests/data/nature_test.json',
                     'ui/static/*',
                     'ui/static/style/*',
                     'ui/static/scripts/*',
                     'ui/home/templates/exsclaim/*',
                     'ui/results/templates/exsclaim/*',
                     'ui/query/templates/exsclaim/*']
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points = {
        'console_scripts': [
            'exsclaim=exsclaim.command_line:run_pipeline',
            'exsclaim_ui=exsclaim.command_line:activate_ui',
            'exsclaim_train=exsclaim.command_line:train',
        ],
    },
    python_requires='>=3.6',
    project_urls={
        'Documentation': 'https://github.com/MaterialEyes/exsclaim/wiki',
        'Source': 'https://github.com/MaterialEyes/exsclaim',
        'Tracker': 'https://github.com/MaterialEyes/exsclaim/issues',
        },
)