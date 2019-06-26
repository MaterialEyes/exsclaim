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

    # Initial caption text
    print("\nInitial caption text:\n%s"%(caption))
    
    # Initial preprocessing of caption text (enforce unicode and resolve known troublesome strings).
    caption = resolve_problem_sequences(caption)
    print("\nResolved caption text:\n%s"%(caption))

    # Create doc and call rule-based matcher on doc
    doc     = nlp(caption)
    matches = matcher(doc)

    # Keywords from webscraper/CDE query
    # Format: {"Main class 1":[synonym1, synonym2, ...], "Main class 2":[synonym1, synonym2, ...], etc."}
    query_kw = {"HAADF-STEM":["HAADF","HAADF-STEM","HAADF–STEM","High-angle annular dark-field","High angle annular dark-field","High-angle annular darkfield","High angle annular darkfield"],\
                "Z-contrast":["Z-contrast","Z–contrast"]}

    if matches != []:
        # Find primary character type (TODO: implement resolve by mixed char type)
        char_type = select_char_delim(nlp,matches,alpha_thresh=0.20)
        
        # Find tokens that satisfy char_type and custom processing pipeline constraints (return # of images implied by caption)
        resolved, num_imgs_implied = resolve_by_char_delim(nlp,doc,matches,char_type)

        print("\nPrimary Char Type: %s"%(char_type))
        print("\nTokens (All): ")
        view_matches(doc,resolved)
    else:
        resolved = []
        num_imgs_implied = 1

    dt, de, dk = associate_caption_text(nlp,doc,resolved,query_kw)

    print("\nPredicted number of images : ",num_imgs_implied)

    print("\nAssociated caption text (by token) : ")
    for k in list(dt.keys()):
        print(k,">"*(9-len(k))," ".join(dt[k]))

    print("\nAssociated caption text (explicit) : ")
    for k in list(de.keys()):
        print(k,">"*(9-len(k))," ".join(de[k]))

    print("\nAssociated caption text (keywords) : ")
    for k in list(dk.keys()):
        print(k,">"*(9-len(k))," ".join(dk[k]))

    print("\n")
    return num_imgs_implied, dt, de, dk 

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

        df['predicted image count']   = [""]*len(df)
        df['caption text (by token)'] = [""]*len(df)
        df['caption text (explicit)'] = [""]*len(df)
        df['caption text (keywords)'] = [""]*len(df)

        for index, row in df.iterrows():
<<<<<<< HEAD
<<<<<<< HEAD
            print("<*>"*35)
            print("Figure ",index)
            caption = row["caption"]
            image_count, dt, de, dk  = interpret_caption(caption,[],nlp,matcher)
            row['predicted image count'] = image_count
            row['caption text (by token)'] = dt
            row['caption text (explicit)'] = de
            row['caption text (keywords)'] = dk
=======
=======
>>>>>>> 6983501... Merge branch 'master' of https://gitlab.com/MaterialEyes/exsclaim
                print("<*>"*35)
                print("Figure ",index)
                caption = row["caption"]
                image_count, dt, de, dk  = interpret_caption(caption,[],nlp,matcher)
                row['predicted image count'] = image_count
                row['caption text (by token)'] = dt
                row['caption text (explicit)'] = de
                row['caption text (keywords)'] = dk
<<<<<<< HEAD
>>>>>>> f78208f... Add htmls from key.csv
=======
>>>>>>> 6983501... Merge branch 'master' of https://gitlab.com/MaterialEyes/exsclaim

        df.to_csv(filename,index=False)