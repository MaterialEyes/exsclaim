Exsclaim JSON Schemas
==========================

The output of the EXSCLAIM! pipeline is an EXSCLAIM JSON which presents data extracted from open, peer-reviewed journals in an organized structure. This structure maps Figure names to their respective Figure JSONs. In each Figure JSON, there is an attribute "Master Images" that maps to Master Image JSON representing each Master Image in the figure. Each Figure JSON also contains metadata and an "unassigned" attribute containing all figure components that have not been organized into a master image. 

Query
---------------------------

- "name" : A string. User defined name of search.
- "journal_family" : A string. Journal family to search from. Current options: "nature", "rsc", "acs", "wiley"
- "maximum_scraped" : An integer. Maximum number of articles to scrape. 
- "sortby" : A string. How to sort or rank which articles to scrape first. Options are "relevant" and "recent"
- "query" : A Search Query JSON.
- "results_dir" : A string (optional). Full path to a directory to put results in. Results directory will be determined as follows: 1. If "results_dir" is supplied, results will go to <results_dir>/<name>. 2. If not supplied, but supplied previously on the current installation, results will go to <most recent <results_dir>/<name>. 3. Otherwise results will be saved to <current working directory>/extracted/<name>
- "save_format" : A list of strings. A list of desired save formats. Options include: 
    - "mongo" for saving to mongo database
    - "csv" for saving to a csv
    - "postgres" for saving to postgres database (also saves to csv)
    - "visualize" for saving a visualization of the resulting subfigures
    - "boxes" for saving figures with bounding boxes drawn.
    - "save_subfigures" for saving all subfigures as images. 
- "mongo_connection" : (optional) A string. URL to mongo database, if one is being used.
- "open" : A boolean. True if only open access results are desired.
- "logging" : A list of at most two strings. If "print" is in the list, logging information will print to stdout (the terminal usually). The other string is, optionally, a file to save logging information to. 


exsclaim (results)
---------------------------

Description
^^^^^^^^^^^^^^^^^^^^^^^^^^

Main result of running the EXSCLAIM Pipeline. A dictionary mapping figure names to their respective Figure JSONs.

Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^

- <figure_name> : An key for each image where its value is the respective Figure JSON

Coordinate JSON
--------------------------

Description
^^^^^^^^^^^^^^^^^^^^^^^^^^

Describes a specific pixel in an image, located "x" pixels from the left of the image and "y" pixels from the top.

Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^

- "x" : An integer. The number of pixels from the left of the image to the specified coordinate.
- "y" : An integer. The number of pixels from the left of the image to the specified coordinate.

Bounding Box JSON
--------------------------

Description
^^^^^^^^^^^^^^^^^^^^^^^^^^

Describes a bounding box (a rectangle) on an image as a list of its four corners. Note that the sides of this bounding box must be parallel to the sides of the image. 

Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^
 
No attributes. Is a list of Coordinate JSONs

Label JSON
--------------------------

Description
^^^^^^^^^^^^^^^^^^^^^^^^^^

Describes either a Subfigure Label or Scale Bar Label by giving its location and contents within the Figure. 

Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^

- "text" : A string. The text of the label. Often something like "(a)" for Subfigure Labels and "10 nm" for Scale Bar Labels. 
- "geometry" : A Bounding Box JSON describing the location of the label.
- "guesses" : (optional) A list of strings. The Text Recognition Model's next best guesses for the text content of the Label

Scale Bar JSON
--------------------------

Description
^^^^^^^^^^^^^^^^^^^^^^^^^^

Describes both a Scale Bar Label and the Scale Bar Line it describes.

Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^

- "label" : A Label JSON. The Scale Bar Label, usually an integer and a unit of measurement. 
- "geometry" : A Bounding Box JSON. The Scale Bar Line.

Child Image JSON
--------------------------

Description
^^^^^^^^^^^^^^^^^^^^^^^^^^

A Child Image contained within the Figure in question described by its location, label, scale bar, and classification. 

Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^

- "label" : (optional) A Label JSON. The Subfigure Label, usually a single letter or number optionally surrounded by parenthesis. 
- "geometry" : A Bounding Box JSON. The location of the Subfigure in the Figure. 
- "scale bar" : (optional) A Scale Bar JSON. Describes the Scale Bar contained within the Subfigure. 
- "classification" : A string. The type of image. One of: Microscopy, Diffraction, Graph, Illustration, Photo, or Unclear. 

Master Image JSON
--------------------------

Description
^^^^^^^^^^^^^^^^^^^^^^^^^^

A Master Image](./definitions#master) (the largest “super” image clearly associated with the neighboring highest order Subfigure Label that captures all images/subfigures the Subfigure label refers to) contained within the Figure. Fully described by its [Child Images, image type, component Scale Bars, Labels, and associated caption. 

Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^
- "label" : (optional) A Label JSON. The Subfigure Label, usually a single letter or number optionally surrounded by parenthesis.
- "geometry" : A Bounding Box JSON. The location of the Master Image in the Figure. 
- "scale bar" : (optional) A Scale Bar JSON. Describes the Scale Bar contained within the Subfigure. 
- "classification" : A string. The type of image. One of: Parent, Microscopy, Diffraction, Graph, Illustration, Photo, or Unclear. 
- "inset images" : (optional) A list of Child Image JSONs](#childimage). All [Inset Images contained within the Master Image. 
- "dependent images" : (optional) A list of Child Image JSONs](#childimage). All [Dependent Images contained within the Master Image. Note: this list should only be filled if "classification" is "Parent".
- "caption" : (optional) A string. The caption chunk from the Figure's article describing the Master Image. 

Figure JSON
--------------------------

Description
^^^^^^^^^^^^^^^^^^^^^^^^^^

Contains all known information about a given Figure](./definitions). Includes metadata, all [Master Images, and all unassigned objects. 

Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^

- "article url" : A string. URL to the article from which the figure came. 
- "journal" : A string. Name of the journal in which the figure appeared. 
- "article title" : A string. Title of the figure's article. 
- "full caption" : A string. The full caption describing the figure. 
- "name" : A string. The name of the figure. The scraper package initializes the name as the article name concatenated with "_fig{i}" where {i} means it was the ith figure scraped from the article. 
- "url" : A string. URL to the figure.
- "open" : A boolean. True if article is open access
- "license" : A string. Text of, or link to article license, if known.
- "master images" : A list of Master Image JSONs](#masterimage). Contains all master images that compose the figure. These are extracted from the "unassigned" attribute in the [cluster module. 
- "unassigned" : An Object JSON](#object). Contains all objects the [object detection package found.  

Object JSON
--------------------------

Description
^^^^^^^^^^^^^^^^^^^^^^^^^^

Contains all objects detected by the object detector](./objects). [Objects](./definitions) include Master Image, Dependent Image, Inset Image, Subfigure Label, Scale Bar Label, Scale Bar Line, and Scale Bar (only included after [cluster. 

Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^

- "Master Image" : A list of Image JSONs. 
- "Inset Image" : A list of Image JSONs. 
- "Dependent Image" : A list of Image JSONs. 
- "Subfigure Label" : A list of Label JSONs. 
- "Scale Bar Label" : A list of Label JSONs.
- "Scale Bar Line" : A list of Bounding Box JSONs.


Image JSON
--------------------------

Description
^^^^^^^^^^^^^^^^^^^^^^^^^^

Describes unassigned Master, Dependent, and Inset Images](./definitions) output from the [object detector

Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^

- "geometry" : A Bounding Box JSON. The location of the Master Image in the Figure. 
- "classification" : A string. The type of image. One of: Parent, Microscopy, Diffraction, Graph, Illustration, Photo, or Unclear.