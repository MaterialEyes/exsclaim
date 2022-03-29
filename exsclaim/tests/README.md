## Test Suite

### Usage

The test suite is divided into validity tests and accuracy tests. The validity tests check whether the code returns correct responses in certain situations. The accuracy tests check how accurate the models used in the pipeline are against certain ground truth, known results.

To run validity tests from the exsclaim/ directory:
```
python3 -m unittest
```

To run accuracy tests:
```
python -m unittest discover -p "*test.py"
```
