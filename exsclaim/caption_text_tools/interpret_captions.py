# -*- coding: utf-8 -*-
import os
import pandas as pd
from caption_tools import *

import spacy
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler

def interpret_caption(caption,keywords,nlp,matcher):
    """
    Matches sequences of caption-relevant tokens to caption text for purposes of segmenting 
    the caption text into segments corresponding to an individual subfigure images. 

    :return: spaCy matcher object (with explicit connections between implied subfigures and tokens), Predicted number of images (subfigures)
    """

    # Initial preprocessing of caption text (enforce unicode and resolve known troublesome strings).
    caption = resolve_problem_sequences(caption)
    print("\nCaption text:\n%s"%(caption[2::]))

    # Create doc and call rule-based matcher on doc
    doc     = nlp(caption)
    matches = matcher(doc)

    query_kw = ["HAADF-STEM","HAADF–STEM","HRTEM","TEM","Z contrast","Z-contrast","annular","darkfield","crystalline","atomic structure","columns","atomic columns","atomic resolution","incoherent"]

    if len(matches) > 0:

        print(matches)
        print(len(matches))
        # Find primary character type (TODO: implement resolve by mixed char type)
        char_type = select_char_delim(nlp,matches,alpha_thresh=0.20)
        
        # Find tokens that satisfy char_type and custom processing pipeline constraints (return # of images implied by caption)
        resolved, num_imgs_implied = resolve_by_char_delim(nlp,doc,matches,char_type)

        print("\nPrimary Char Type: %s"%(char_type))
        print("\nTokens (All): ")
        view_matches(doc,resolved)

        d = {}
        d = associate_caption_text(nlp,doc,resolved,query_kw)

    else:

        print("No Tokens Found!")
        resolved = []
        num_imgs_implied = 1
        d = {}
        d["None"] = caption

    print("\nPredicted number of images : ",num_imgs_implied)


    print("\nAssociated caption text (by token) : ")
    for k in list(d.keys()):
        print(k,">"*(12-len(k)),d[k])

    print("\n")
    return num_imgs_implied,d

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

nlp, matcher = custom_nlp_pipeline(offsets,position_keys,separations,char_types,char_nums,custom_patterns)

directory_name = os.getcwd()+"/csv"
directory = os.fsencode(directory_name)

for root, dirs, files in os.walk(directory):
    for file in files:
        filename = os.fsdecode(os.path.join(root, file))
        df = pd.read_csv(filename)

        for index, row in df.iterrows():

            # if row["num subfigs"] != 1000 and row["num subfigs"] != -99:
         
            print("<*>"*35)
            print("Figure ",index)
            caption = ". "+row["caption"]
            image_count, d = interpret_caption(caption,[],nlp,matcher)
            

                # if image_count != row["num subfigs"]:
                #     print("Expected number of images: ",row["num subfigs"])
                #     break
