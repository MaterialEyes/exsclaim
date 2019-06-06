# -*- coding: utf-8 -*-
import os
import pandas as pd

import spacy
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler

from caption_tools import *

def scientific_caption_matcher(caption,nlp,matcher):
    """
    Matches sequences of caption-relevant tokens to caption text for purposes of segmenting 
    the caption text into segments corresponding to an individual subfigure images. 

    :return: spaCy matcher object (with explicit connections between implied subfigures and tokens), Predicted number of images (subfigures)
    """
    # Initial preprocessing of caption text (enforce unicode and resolve known troublesome strings).
    caption = resolve_problem_sequences(caption)
    doc = nlp(caption)
    matches = matcher(doc)

    if len(matches) > 0:
        char_type = select_char_delim(nlp,matches,alpha_thresh=0.20)
        print("\nChar Type: %s"%(char_type))
        resolved, image_count = resolve_by_char_delim(nlp,doc,matches,char_type)
        print("\nTokens (All): ")
        view_matches(doc,resolved)
    else:
        print("No Tokens Found!")
        resolved = []
        image_count = 1

    print("\nPredicted number of images : ",image_count)
    
    # UNDER DEVELOPMENT
    # associate_caption_text(nlp,doc,resolved)

    return resolved,image_count




# Description of parameters used to create caption pattern collection:
# - offsets: punctuation used to set off characters that are explanatory (i.e. a parenthesis)
# - position_keys : position of the offset character: 0-before, 1-after, 2-both
# - separations: delimiter between characters within the offsets
# - char_types: the character type (letter-> alpha or ALPHA (capitalized), digit-> number ... etc)
# - char_nums: number of consecutive characters between delimeters
# - custom_pattern: a list of tuples containing any custom patterns (from observation)

offsets         =  ['parenthesis','colon']
position_keys   =  [0,1,2]
separations     =  ['comma','dash','and','none']
char_types      =  ['alpha','ALPHA','digit','roman','position']
char_nums       =  range(1,9)
custom_patterns =  [{'label': "parenthesis_02_none_ALPHA_02", 'pattern': [{'ORTH': "("},{'TEXT': {"REGEX":'[A-Z]{1}\d{1}'}},{'ORTH': ")"}]},\
                    {'label': "parenthesis_02_none_alpha_02", 'pattern': [{'ORTH': "("},{'TEXT': {"REGEX":'[a-z]{1}\d{1}'}},{'ORTH': ")"}]}]

# Create caption specific patterns from inputs
caption_patterns = caption_pattern_collection(offsets,position_keys,separations,char_types,char_nums)
caption_patterns.extend(custom_patterns)

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
matcher = load_patterns_to_matcher(matcher,caption_patterns)

ruler = EntityRuler(nlp, patterns=caption_patterns)
nlp.add_pipe(ruler, before="ner")

#Path to .csv file containing caption/image name information
csv_path = os.getcwd()+"/high_angle_annular_dark_field_scanning_transmission_electron_microscopy_imgs.csv"
# csv_path = os.getcwd()+"/material_eyes_150419.csv"

df = pd.read_csv(csv_path)

for index, row in df.iterrows():

    if row["num subfigs"] != 1000 and row["num subfigs"] != -99:
 
        caption = row["caption"]
        print("\n"+"<*>"*30+"\n"+"Caption [%d]: "%(index))
        print(caption)

        subfigure_tokens,image_count = scientific_caption_matcher(caption,nlp,matcher)

        if image_count != row["num subfigs"]:
            print("Expected number of images: ",row["num subfigs"])
            break

        # j=j78

