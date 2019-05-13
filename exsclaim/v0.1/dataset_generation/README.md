# Dataset Generation
This directory contains code for the generation of artificial datasets

## dataset_generator.py
This file is made to generate images with randomly placed text and corresponding
text files (with the same name as corresponding images) with the location of each
piece of text. This is in the format to retrain a model for EAST described [here](https://github.com/argman/EAST).
To generate images use the following interface:
```
$ python dataset_generator.py --number (-n) [--samples_images (-d) | --output (-o)]
    sample_images: path to directory containing no text (default is "no_text")
    number: number of examples to be generated
    output: path to desired output directory (default is "generated_examples")
```
