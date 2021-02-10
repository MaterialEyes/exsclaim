## EXSCLAIM Module

This directory contains the relevant code for the EXSCLAIM module, organized as follows:

 - pipeline.py: Code for Pipeline class, which runs ExsclaimTool subclasses on Query JSONs.
 - tool.py: Defines base ExsclaimTool class and CaptionDistributor and JournalScraper subclasses. Each ExsclaimTool class is initialized with a Query JSON and has methods to run the tool, update the exsclaim.json, and load any necessary models. 
 - figure.py: Defines FigureSeparator, a subclass of ExsclaimTool. Has methods to separate, classify, and read the labels of subfigures and to determine the scale of subfigures.
 - caption.py: Defines useful functions used by CaptionDistributor.
 - journal.py: Defines the JournalFamily class. Nature and ACS subfamilies are defined and other journal families can be added and used by JournalScraper.
 - figures/: Additional modules for defining and training figure models.
 - captions/: Additional modules for defining and training caption models.
 - tests/: test files and sample data to check results.
 - utilities/: Modules with functions that are useful across several modules