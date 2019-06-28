# -*- coding: utf-8 -*-
import sys
import os
import re
import ast
import yaml
import string
import pickle
import difflib
import itertools
import numpy as np
import collections
from itertools import chain, zip_longest
from difflib import SequenceMatcher

import spacy
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler

# Helper functions for customizing spaCy nlp processing pipeline and matchers
def custom_nlp_pipeline(offsets,position_keys,separations,char_types,char_nums,custom_patterns):
    """
    Add custom components to spaCy nlp processing pipeline

    # Description of parameters used to create caption pattern collection:
    :param: offsets: punctuation used to set off characters that are explanatory (i.e. a parenthesis)
    :param: position_keys : position of the offset character: 0-before, 1-after, 2-both
    :param: separations: delimiter between characters within the offsets
    :param: char_types: the character type (letter-> alpha or ALPHA (capitalized), digit-> number ... etc)
    :param: char_nums: number of consecutive characters between delimeters
    :param: custom_pattern: a list of tuples containing any custom patterns (from observation)

    :return: spaCy nlp processing pipeline, spaCy rule-based Matcher
    """
    # Create caption specific patterns from inputs
    caption_patterns = caption_pattern_collection(offsets,position_keys,separations,char_types,char_nums)
    caption_patterns.extend(custom_patterns)

    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)
    matcher = load_patterns_to_matcher(matcher,caption_patterns)
    ruler = EntityRuler(nlp, patterns=caption_patterns)
    nlp.add_pipe(ruler, before="ner")

    return nlp, matcher

def caption_parse_rules(offset,position_key,separate,char_type,char_num):
    """
    Create matching rules and based on custom characteristics of caption text described in input vars

    :param offset: characters used to set off material that is amplifying or explanatory
    :param position_key : position of the offset character: 0-before, 1-after, 2-both
    :param separation: delimiter between characters within the offsets
    :param char_type: the character type (letter-> alpha or ALPHA (capitalized), digit-> number ... etc)
    :param char_num: number of consecutive characters between delimeters

    :return: A spaCy Match pattern. A pattern consists of a list of dicts, where each dict describes a token.
    """
    roman_keywords = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii', 'xiii', 'xiv', 'xv']
    position_keywords =["above","after","around","before","beginning","behind","below","beside","between",\
                        "bottom","center","down","end","far","finish","front","in","inside","lower","left","middle","near","next to",\
                        "off","on","out","outside","over","right","start","through","top","under","up","upper","upside down"]

    # Get subfigure label offset char (i.e parenthesis mark off test specific to an individual image)
    offset_choices = {'parenthesis':'()','bracket':'[]','colon':'::','period':'..'}
    offset_delimiter = offset_choices.get(offset)

    # Get subfigure label separation char (i.e this punctuation separated consective chars)
    separate_choices = {'comma':',','dash':'–','colon':':','period':'.','and':'and','none':'none'}
    separate_delimiter = separate_choices.get(separate)

    # Get character type (input to SHAPE), or list of characters from a predefined vocabulary
    character_choices = {'alpha':'x','ALPHA':'X','digit':'d','roman':roman_keywords,'position':position_keywords}
    character_delimiter = character_choices.get(char_type)

    # Get keywords associcated with a given char_type
    keywords_choices = {'roman':roman_keywords,'position':position_keywords}
    keywords = keywords_choices.get(char_type)

    if not keywords:
        if separate_delimiter not in ['none','and']:
            body = [{'SHAPE': character_delimiter},{'ORTH': separate_delimiter}]*(char_num-2) + [{'SHAPE': character_delimiter},{'ORTH': separate_delimiter, 'OP': '?'},{'ORTH': 'and', 'OP': '?'}]*2
        else:
            body = [{'SHAPE': character_delimiter},{'ORTH': separate_delimiter}]*(char_num)
    else:
        if separate_delimiter  not in ['none','and']:
            body = [{"LOWER": {"IN": keywords}},{'ORTH': separate_delimiter}]*(char_num-2) + [{"LOWER": {"IN": keywords}},{'ORTH': separate_delimiter, 'OP': '?'},{'ORTH': 'and', 'OP': '?'}]*2
        else:
            body = [{"LOWER": {"IN": keywords}},{'ORTH': separate_delimiter}]*(char_num)

    if position_key == 0:
        return [{'ORTH': offset_delimiter[0]}]+body[:-1]
    elif position_key == 1:
        return body[:-1]+[{'ORTH': offset_delimiter[1]}]
    elif position_key == 2:
        return [{'ORTH': offset_delimiter[0]}]+body[:-1]+[{'ORTH': offset_delimiter[1]}]
    else:
        raise ValueError('The offset delimiter must appear to the left (0), right (1), or both sides (2) of the character!') 

def caption_pattern_collection(offsets,position_keys,separations,char_types,char_nums):
    """
    A wrapper around caption_parse_rules to create a collection of rules and descriptions based on input lists

    :return: List of tuples representing the name of each token (rule) with a list of dicts, where each dict describes a token.
    """
    full = []
    descriptions, patterns = [], []
    for off in offsets:
        for pk in position_keys:
            for sep in separations:
                for ct in char_types:
                    for cn in char_nums:
                        if cn == 1:
                            rules = caption_parse_rules(off,pk,sep,ct,cn)
                            description=str(off)+"_"+str(pk).zfill(2)+"_"+"none"+"_"+str(ct)+"_"+str(cn).zfill(2)
                        else:
                            rules = caption_parse_rules(off,pk,sep,ct,cn)
                            description=str(off)+"_"+str(pk).zfill(2)+"_"+str(sep)+"_"+str(ct)+"_"+str(cn).zfill(2)
                        
                        full.append({"label":description, "pattern":rules})
                        descriptions.append(description)
                        patterns.append(rules)
    return full

def load_patterns_to_matcher(matcher,patterns):
    """
    Load custom patterns (rules) to a spaCy matcher

    :param matcher: The spaCy matcher object
    :param patterns: The rules (readable by spaCy matcher)

    :return: The matcher containing the a new set of rules
    """
    for pattern in patterns:
        matcher.add(pattern['label'],None,pattern['pattern'])
    return matcher

#  ___                               ____                  _   _             
# |_ _|_ __ ___   __ _  __ _  ___   / ___|___  _   _ _ __ | |_(_)_ __   __ _ 
#  | || '_ ` _ \ / _` |/ _` |/ _ \ | |   / _ \| | | | '_ \| __| | '_ \ / _` |
#  | || | | | | | (_| | (_| |  __/ | |__| (_) | |_| | | | | |_| | | | | (_| |
# |___|_| |_| |_|\__,_|\__, |\___|  \____\___/ \__,_|_| |_|\__|_|_| |_|\__, |
#                      |___/                                           |___/ 

def make_unicode(str_text: str) -> str:
    """
    Ensures UTF-8 can be used in strings

    :param str_text: A string of text

    :return: Plain string UTF decoded
    """
    if type(str_text) != str:
        str_text =  str_text.decode('utf-8')
        return str_text
    else:
        return str_text

def remove_digits(text):
    """
    Removes digits from string of text (or list of strings)

    :param str_text: A string of text

    :return: Plain string with numerical digits removed
    """
    return ''.join([i for i in text if not i.isdigit()])

def is_no_overlap(l1,l2) -> bool:
    """
    Determines if two lists share any common elements

    :param l1: List 1
    :param l2: List 2

    :return: A bool determining if lists overlap at all
    """
    return len(set(l1).intersection(set(l2))) == 0

def flatten(items: list) -> list:
    """
    Yield items from any nested iterable; 

    :param items: A nested iterable (list)

    :return: A flattened list
    """
    for x in items:
        if isinstance(x, collections.Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def fullstop_delimeter(str_text: str):
    """
    Identifies the primary char used as full-stop for sentence delimiting.
    NOTE: This is a naive frequency-based implementation without NLP.

    :param str_text: A string of text

    :return A char of the appropriate delimeter
    """
    return ("; ",". ")[str_text.count(". ")>=str_text.count("; ")]

def resolve_semicolon_dependent_clause(str_text: str, trigger_pair):
    """
    Attempts to find scenarios where authors separated dependent clauses
    with semi-colons and replace the affected semi-colons with commas. 

    :param str_text: A string of text

    :return A resolved string of text
    """
    fs = fullstop_delimeter(str_text)

    if fs == "; ":
        sidx,eidx = (-1,-1)
        chunks = str_text.split(fs)
        for i in range(len(chunks)):
            if trigger_pair[0] in chunks[i] and sidx < 0:
                sidx = i
            if trigger_pair[1] in chunks[i] and sidx >= 0:
                eidx = i+1
        if eidx == -1:
            eidx = len(chunks)+1

        chunks_critical = str_text.split(fs)[sidx:eidx]

        new_text = ""
        for chunk in chunks[:-1]:
            if chunk in chunks_critical:
                new_text += chunk+", "
            else:
                new_text += chunk+"; "
        new_text += chunks[-1]
    else:
        new_text = str_text

    return new_text

def resolve_problem_sequences(str_text: str) -> str:
    """
    Resolves known problematic string sequences

    :param str_text: A string of text

    :return: Plain string UTF decoded with problem sequences resolved
    """
    str_text = make_unicode(str_text)

    # Resolve known problematic string sequences
    string_subs = [[")-(","-"],[")–(","–"],[":(",": ("],["):",") :"]]
    for subs in string_subs:
        str_text = " ".join([a.replace(subs[0],subs[1]) for a in str_text.split(" ")])

    str_text = re.sub(' +', ' ',str_text)

    str_text = resolve_semicolon_dependent_clause(str_text,["images of"," ."])
    str_text = resolve_semicolon_dependent_clause(str_text,["with precipitates"," ."])

    # Custom substitution of redundant phrases to assist in caption correspondence. 
    str_text = str_text.replace(". From left to right: (a)",": (a)")
    str_text = str_text.replace("of image (","shown in image (")
    str_text = str_text.replace(") and (",",")
    str_text = str_text.replace("(a) :",": (a)")
    str_text = str_text.replace("1 –","1 -")
    str_text = str_text.replace("2 –","2 -")
    str_text = str_text.replace("3 –","3 -")
    str_text = str_text.replace("N-C.","N-C .")

    return str_text

def view_matches(doc,matches):
    """
    Tool to view the matches and their associated properties 

    :param doc: The spaCy tokenized caption text
    :param matches: The spaCy match object (list of match tuples)

    :return: Null
    """
    for match_id, string_id, start, end, token, implied in matches:
        span = doc[start:end]  # The matched span
        print(string_id, start, end, token,implied)

def letter_to_int(letter: str) -> int:
    """
    Finds letter number position in alphabet (1-26)

    :param input: A letter (string)

    :return: Position in the alphabet
    """
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    return alphabet.index(letter.lower()) + 1

def roman_to_int(letter: str) -> int:
    """
    Finds letter number position in roman numeral list

    :param input: A letter (string)

    :return: Position in the roman numeral list (decoding of roman numeral)
    """
    roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', \
             'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', \
             'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', \
             'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', \
             'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX', 'XXXI',\
             'XXXII', 'XXXIII', 'XXXIV', 'XXXV', 'XXXVI', \
             'XXXVII', 'XXXVIII', 'XXXIX', 'XL', 'XLI', 'XLII',\
             'XLIII', 'XLIV', 'XLV', 'XLVI', 'XLVII', 'XLVIII', \
             'XLIX', 'L']
    return roman_numerals.index(letter.upper())+1

def interpolate_chars(start: str, end: str, char_type: str) -> str:
    """
    Appropriate interpolation between endpoints given the context of the char_type

    :param start: Starting letter, digit, or roman numeral (string or int)
    :param end: Ending letter, digit, or roman numeral (string or int)
    :param char_type: The character type (alpha,digit, or roman)

    :return: A comma delimited list containing the proper interpolation between endpoints 
    """
    roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', \
         'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', \
         'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', \
         'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', \
         'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX', 'XXXI',\
         'XXXII', 'XXXIII', 'XXXIV', 'XXXV', 'XXXVI', \
         'XXXVII', 'XXXVIII', 'XXXIX', 'XL', 'XLI', 'XLII',\
         'XLIII', 'XLIV', 'XLV', 'XLVI', 'XLVII', 'XLVIII', \
         'XLIX', 'L']

    if char_type.lower() == 'alpha' or char_type == 'ALPHA':
    
        st = letter_to_int(start)-1
        ed = letter_to_int(end)-1

        if start == start.upper():
            return ",".join(string.ascii_uppercase[st:ed+1])
        else:
            return ",".join(string.ascii_lowercase[st:ed+1])

    elif char_type.lower() == 'roman':

        st = roman_to_int(start)-1
        ed = roman_to_int(end)-1

        if start == start.upper():
            return ",".join(roman_numerals[st:ed+1])
        else:
            return ",".join([a.lower() for a in roman_numerals[st:ed+1]])

    elif char_type.lower() == 'digit':
        return ",".join([str(a) for a in np.arange(start,end+1)])

def implied_chars(text: str, char_type: str) -> str:
    """
    Finds the appropriate full string representation by implying a sequence of alphabetically or numerically connected characters (when relevant)

    :param text: The starting text string 
    :param char_type: The character type (alpha,digit, or roman)

    :return: A comma delimited string containing the implied full sequence of characters
    """
    position_keywords =["above","after","around","before","beginning","behind","below","beside","between",\
                    "bottom","center","down","end","far","finish","front","in","inside","lower","left","middle","near","next to",\
                    "off","on","out","outside","over","right","start","through","top","under","up","upper","upside down"]

    text = (text.strip("(").strip(")").strip(" "))
    text = text.replace("and"," , ").replace(" ","").replace("-","–").replace(":","")

    discrete_tokens = [","]
    continuous_tokens = ["–"]

    token_list = discrete_tokens + continuous_tokens
    token_key = ("|").join(token_list)

    chars = re.split(token_key,text)
    delims = [a for a in text if a in token_list]

    # Interleave chars with delims
    text_list = [x for x in chain.from_iterable(zip_longest(chars,delims)) if x is not None]
    text_list = [remove_digits(a) for a in text_list]

    if char_type == 'position':
        return ",".join([a for a in text_list if a.lower() in position_keywords])

    if len(text_list) == 1:
        return text_list[0]

    IC = ""
    for i in np.arange(0,len(text_list)-1,2):
        if text_list[i+1] in discrete_tokens:
            DT = ",".join([text_list[i],text_list[i+2]])
        else:
            DT = ""
        if text_list[i+1] in continuous_tokens:
            CT = interpolate_chars(text_list[i],text_list[i+2],char_type)
        else:
            CT = ""

        IC += DT+CT+","

    return ",".join([a for a in np.unique(IC[:-1].split(",")) if len(a)>0])

def isrelated(test: str, fixed: list, connections: list) -> bool:
    """
    Determines if test char (alpha) is related to fixed list of alpha chars.
    Related means that test is contained in the span of fixed, or attached via a connections list, or is immediately adjacent.
    Letters with connections that contain numbers in them are considered units and thus not related.

    :param fixed: A list of relevant alpha chars
    :param test: A string "abc" that might be related to fixed
    :param connections: A list of lists where each list represents a group or connected chars

    :return: A decision on whether it is related or not
    """
    test_id  = letter_to_int(test.lower())
    fixed    = sorted(list(np.unique([a for a in fixed if a not in test])))
    fixed_id = letter_to_int(fixed[-1].lower())

    isconnected = 0
    for entry in connections:
        if test in entry:
            if len([value for value in entry if value in fixed])>0:
                isconnected = 1

    isnumeric = 0
    for entry in connections:
        if test in entry:
            if len([s for s in entry if s.isdigit()]) > 0:
                isnumeric = 1

    isadjacent = 0
    for entry in fixed:
        if np.abs(test_id-letter_to_int(entry.lower())) == 1:
            isadjacent = 1

    if fixed_id > test_id or (isconnected and not isnumeric) or isadjacent:
        return True
    else:
        return False

def build_connection_lists(str_text: str) -> list:
    """
    Finds all segements of text enclosed in parenthesis and groups each segment into a list

    :param: str_text: A string of text

    :return: A list of lists where each list represents a group text chars related by parenthesis
    """
    connection_lists = []

    string_subs = [[")–(","–"],[")-(","–"]] # Resolve known problematic string sequences
    for subs in string_subs:
        str_text = " ".join([a.replace(subs[0],subs[1]) for a in str_text.split(" ")])

    str_text = re.findall('\(.*?\)',str_text)
    
    for entry in str_text:
        elem = str(re.sub(r'[^\w\s]',' ',entry)).split(" ")
        connection_lists.append([a for a in elem if a not in ['and','']])

    return connection_lists

def select_char_delim(nlp,matches,alpha_thresh=0.5) -> str:
    """
    Determine which char_type is functioning as a delimiter between segments of text describing the individual subfigures. 

    :param nlp: Base NLP model (spaCy) 
    :param matches: The spaCy match object (list of match tuples)
    :alpha_threshold: Accept "alpha" as char_type when num_alpha_matches/num_most_freq_match_type > alpha_thresh

    :return: The primary char_type in the list of matches
    """
    char_types = []
    for m in matches:
        char_types.append(nlp.vocab.strings[m[0]].split("_")[3])

    c = collections.Counter(char_types)

    if char_types == []:
        return None

    first_char = char_types[0]
    most_freq_char = c.most_common(1)[0][0]
    max_alpha = np.max([c['alpha'],c['ALPHA']])

    alpha_most_common_ratio = float(max_alpha)/c.most_common(1)[0][1]

    if alpha_most_common_ratio>= alpha_thresh:
        keep_char = 'ALPHA' if np.argmax([c['alpha'],c['ALPHA']]) else 'alpha'
    elif most_freq_char == 'roman':
        keep_char = most_freq_char
    else:
        keep_char = first_char

    return keep_char

def resolve_by_char_delim(nlp,doc,matches,char_type):
    """
    Filter out redundant subsegments of related tokens

    :param nlp: Base NLP model (spaCy) 
    :param doc: The spaCy tokenized caption text
    :matches: The list of tuples containing characters of interest
    :param char_type: The character type (alpha, digit, or roman)

    :return: Resolved (related tokens combined into single token) list of tuples based on char_type (i.e. [" (a, "," (a, b "," (a, b) "] --> "(a, b)") 
    """
    chemical_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', \
                         'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti',\
                         'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', \
                         'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', \
                         'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', \
                         'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', \
                         'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', \
                         'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', \
                         'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', \
                         'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', \
                         'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    ambiguous_tokens = ["(a"]

    # Filter the matches by the given char_type (TODO: accomodate a mixed char_type scenario)
    # At this point, the filtering procedure has not been able to perfectly localize the token of interest. 
    # The following steps identify the critical indices corresponding the the most complete tokens
    char_filt  = [a for a in matches if nlp.vocab.strings[a[0]].split("_")[3] == char_type]

    start_end_segments = [list(np.arange(a[1],a[2])) for a in char_filt]
    start_idxs = [i for i, x in enumerate([True]+[is_no_overlap(start_end_segments[i],start_end_segments[i+1]) for i in range(len(start_end_segments)-1)]) if x]+[len(char_filt)]

    size_lists = [[np.abs(a[1]-a[2]) for a in char_filt[start_idxs[i]:start_idxs[i+1]]] for i in range(len(start_idxs)-1)]
    offsets = [np.argmax(a) for a in size_lists]
    critical_idxs = list(np.array(start_idxs[0:-1])+np.array(offsets))

    # Filter out redundant subsegments of related tokens
    localized = [char_filt[i] for i in critical_idxs] 

    # Filter out ambiguous token (i.e. FP tokens)
    localized = [a for a in localized if doc[a[1]:a[2]].text not in ambiguous_tokens]

    # Create a list that contains all the chars in the localized segments
    fixed = sorted(np.unique(list(flatten([list(a) for a in [implied_chars(doc[entry[1]:entry[2]].text,char_type).split(",") for entry in localized]]))))
    fixed = [a for a in fixed if a.title() not in [c for c in chemical_elements if len(c)>1]]

    # Establish the connection points from the caption (i.e. a–d implies that "d" is connected to "a")
    connections = build_connection_lists(doc.text)

    # Determine if the selected chars function as delimiters in the caption (i.e. roman numeral "i", versus subfigure label "i")
    # Also filter out any 2 letter chemical elements that have made it through
    # If the token belongs in the caption, create an array that explicitly shows what it implies
    implied_full, in_caption = [], [] 
    for match_id, start, end in localized:  
        string_id = nlp.vocab.strings[match_id]
        related = 1
        if char_type in ['alpha','ALPHA'] and len(localized)>1:
            for letter in implied_chars(doc[start:end].text,char_type).split(","):
                if letter.title() in [c for c in chemical_elements if len(c)>1]:
                    related = 0
                elif isrelated(letter,fixed,connections) == False:
                    related = 0
        if related:
            implied = implied_chars(doc[start:end].text,char_type).replace(":","").replace("(","").replace(")","").split(",")
            implied_full.extend(implied)
            in_caption.append((match_id,string_id,start,end,doc[start:end].text,implied))

    return in_caption, len(np.unique(implied_full))

#  ___                                 _            _                                  _   
# |_ _|_ __ ___   __ _  __ _  ___     / \   ___ ___(_) __ _ _ __  _ __ ___   ___ _ __ | |_ 
#  | || '_ ` _ \ / _` |/ _` |/ _ \   / _ \ / __/ __| |/ _` | '_ \| '_ ` _ \ / _ \ '_ \| __|
#  | || | | | | | (_| | (_| |  __/  / ___ \\__ \__ \ | (_| | | | | | | | | |  __/ | | | |_ 
# |___|_| |_| |_|\__,_|\__, |\___| /_/   \_\___/___/_|\__, |_| |_|_| |_| |_|\___|_| |_|\__|
#                      |___/                          |___/                                

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def split_list(l):
    """
    Groups neighboring list entries into single list.
    List entries are considered neighbors as long as they do not border "*" or "!"

    :param l: A list of strings

    :return: A list of lists where neighbors are grouped into list
    """
    master = []
    sub = []
    for a in l:
        if a[0] != "*" and a[0] != "!":
            sub.append(a)
        else:
            master.append(sub)
            master.append([(a,a)])
            sub = []
    if sub != []:
        master.append(sub)

    return master

def does_segment_exist(seq, sublist):
    """
    :return: XXXXXX
    """
    first = sublist[0]
    i = 0
    n = len(sublist)
    while True:
        try:
            i = seq.index(first, i)
        except ValueError:
            return (-1,-1)
        if sublist == seq[i:i+n]:
            return (i,i+n)
        i += 1

def fill_gaps(l):
    li = l.copy()
    # first fill in gaps
    cnt = 0
    for a,b in l:
        if a > cnt:
            l.insert(cnt, tuple((cnt, 1)))
        cnt += 1

    return l[l.index(li[0])::]

def filter_by_tokenized_search_patterns(l1,l2,d):
    """
    Filter out redundant subsegments of related tokens

    :param l1: tokenized_caption [(token_1,POS_tag_1,index_1),(token_2,POS_tag_2,index_2)...], \
               i.e. [('(d)', 'CAP', 1), ('Exemplary', 'NC', 2), ('averaged', 'VBD', 3), ('distances', 'NC', 4) ...]
    :param l2: token_pattern
    :param d : dictionary (keys define master token)

    :return: dictionary 
    """
    # Parse token_pattern by "*" and "!" (These are wildcard/stop characters in the string search)
    split_l2 = split_list(l2)

    # Iteratively move the starting point of the search 
    for tt in range(len(l1)):

        l1_beg,l1_end = tt,len(l1)

        intervals  = []

        # Search for first unbroken subsequence of token_pattern in the caption
        for frag in split_l2:
            
            # Master caption sequence with the current starting point
            master    =  l1[l1_beg:l1_end:1]
            
            # Get all the POS tokens in the token_pattern subsequence
            frag_pos  =  [a[0] for a in frag]
            
            # Get all the POS tokens in the caption sequence
            filt_pos  =  [a[1] for a in master]
            
            if frag_pos[0] != "*" and frag_pos[0] != "!":
                
                # Get start/end points for existence of token_pattern subsequence in the caption sequence
                seg_beg,seg_end = does_segment_exist(filt_pos, frag_pos)

                if seg_beg == -1: # Subsequence does not exist
                    break
                else:
                    l1_beg = master[seg_end-1][2]+1 # Resume search next time through master at end of the critical slice
                    
                    # Get the absolute indices of the critical tokens in master
                    critical_idxs = [a[2] for a in master[seg_beg:seg_end]]
                    
                    # Create list of tuples for critical tokens - tuples contain absolute index and T/F inclusion flag (i.e. [(120, 1), (121, 1)])             
                    idxs_switch = list(zip(critical_idxs,[a[1] for a in frag]))
                    intervals.extend(idxs_switch)
            else:
                # If wildcard/stop token is a stop, append to current list to indicate break
                if frag_pos[0] == "!":
                    intervals.extend(frag_pos) # Token to indicate break!

        # All subsequences of the token_pattern must have been found (i.e no seg_beg = -1) to continue
        # If subsequence found...
        if seg_beg > -1:

            # Group contiguous intervals and fill implied gaps 
            if intervals != []:
                intervals_extended = []
                for sl in split_list(intervals):
                    if sl[0] != ('!', '!') and len(sl)>0:
                        intervals_extended.extend(fill_gaps(sl))

            # In an effort to eliminate solutions with multple "CAP"s and with full-stops in the middle...
            cap_count   = [l1[a[0]][1] for a in intervals_extended if a[1] == 1].count('CAP')
            
            # Method #1 (lenient)- only search for stops among characters that are not ignored!
            # midfs_count = [l1[a[0]][1] for a in intervals_extended[0:-1]].count('.')

            # Method #2 (strict) -  search for stops occuring between endpoints
            pos_list = [l1[a][1] for a in list(np.arange(intervals_extended[0][0],intervals_extended[-1][0]))]
            midfs_count = pos_list.count('.') + pos_list.count(';')
            
            # Count number of internal references made (cue to include additional CAP separate from the master CAP)
            # i.e. we want to keep (b) in this situation: (c) HAADF STEM image of the the region indicated in (b). 
            iref_count  = [l1[a[0]][1] for a in intervals_extended if a[1] == 1].count('IR')

            # Retain only sentences with CAPS after the IR (there are instances where this fails, but should fail conservatively)
            iref_valid = True
            if iref_count:
                reversed_pos_list = [l1[a[0]][1] for a in reversed(intervals_extended) if a[1] == 1]
                if reversed_pos_list.index("IR") < reversed_pos_list.index("CAP"):
                    iref_valid = False

            # Issue #1:
            # Full stop punctuation in the middle of the line.
            issue_1 = midfs_count != 0

            # Issue #2:
            # Multiple CAP refs without an IR indicator, 
            issue_2 = (cap_count >1 and iref_count==0)

            # Issue #3:
            # More than 2 CAP refs for a single IR indicator
            issue_3 = (cap_count > 2 and iref_count==1)

            # Issue #4:
            # Multiple CAP refs with an IR indicator, but invalid IR
            issue_4 = (cap_count >1 and iref_count==1 and iref_valid==False)

            if not (issue_1 or issue_2 or issue_3 or issue_4):
                # Assign text string to governing CAP
                cap_str = [l1[a[0]][0] for a in intervals_extended if a[1]==1 and l1[a[0]][1]=='CAP'][0]
                # Convert sequence of match tokens to text string
                match = " ".join([l1[a[0]][0] for a in intervals_extended if a[1]==1])
                # Assign key (determined by governing CAP) to the matched text string. 
                d[cap_str].append(match.replace("( ","(").replace(" )",")"))

    # Retain only unique values (entries)
    for k in list(d.keys()):
        d[k] = list(np.unique(d[k]))
    enablePrint()
    
    return d

def spans_from_list(doc,keywords):
    return tuple([doc[i:i+1] for i in range(len(doc)) if doc[i:i+1].text in keywords])

def spans_from_spans(doc,spans):
    return [doc[i:i+1] for i in range(len(doc)) if doc[i:i+1] in spans]

def custom_retoken(doc,custom_spans,custom_tag):
    with doc.retokenize() as retokenizer:
        for span in custom_spans:
            retokenizer.merge(span,attrs={"TAG":custom_tag})
    return doc

def nounchunk_retoken(doc,remove_spans):
    # Combine tokens to form noun chunks, and remove already defined tokens
    noun_chunk_spans = []
    for chunk in doc.noun_chunks:
        spans_to_remove = spans_from_spans(chunk,remove_spans)
        if(spans_to_remove):
            # Find id's of CAP tokens and ',' to remove from noun chunks. 
            rm_ids = [chunk.start+i for i in range(chunk.end-chunk.start) if doc[chunk.start+i].text in [a.text for a in spans_to_remove] or doc[chunk.start+i].text == ","]
            for n in range(len(rm_ids)):
                if n == len(rm_ids)-1:
                    rev = doc[rm_ids[n]+1:chunk.end]
                else:
                    rev = doc[rm_ids[n]+1:rm_ids[n+1]]
                if len(rev)>0:
                    noun_chunk_spans.append(rev)
        else:
            noun_chunk_spans.append(chunk)

    # Retokenize doc tagging the noun chunks (with caption tokens removed)
    with doc.retokenize() as retokenizer:
        for span in noun_chunk_spans:
            retokenizer.merge(span,attrs={"TAG":"NC"})

    return doc

def is_subset(a,b,tol=0):
    # See if b is subset of a
    s = SequenceMatcher(None, a, b)
    match_elems = []
    for block in s.get_matching_blocks():
        _ , _ , match = block 
        match_elems.append(match)
    return np.max(match_elems) >= len(b)-np.floor((tol*len(b)))

def is_beginning_mismatch(a,b):
    # See if string mismatch occurs with first char
    s = SequenceMatcher(None, a, b)
    match_elems = []
    for block in s.get_matching_blocks():
        start_match, _, _ = block
        if start_match == 0:
            return False
        else:
            return True

def associated_dictionary_entry(d,text_str):
    keyword_list = list(flatten([d[a] for a in d]))
    dict_entries = list(np.unique([a for a in keyword_list if is_subset(text_str.lower(),a.lower(),tol=0.25)]))
    if dict_entries != []:
        return list(np.unique([find_dictionary_key(d,a)for a in dict_entries]))
    else:
        return []

def find_dictionary_key(d,test_string):
    return [a for a in list(d) if test_string.lower() in [b.lower() for b in d[a]]][0]


def cleanup_strings(str_list,token):
    
    # Remove token from string only if it is at the beginning
    str_list_notok = [a.replace(token,"",1) for a in str_list]

    str_resolved = []
    for i in range(len(str_list)):
        if is_beginning_mismatch(str_list[i],str_list_notok[i]):
            str_resolved.append(str_list_notok[i])
        else:
            str_resolved.append(str_list[i])

    # Remove superfluous spaces and punctuation marks
    return [text.strip(",").strip(".").strip(";").strip(" ").replace(" ,",",")+"." for text in str_resolved]

def longest_unrelated_entries(str_list,ratio):
    str_list = list(np.unique(str_list))
    remove_list = []
    for pair in itertools.combinations(str_list, r=2):
        v,w = pair
        s = difflib.SequenceMatcher(None, v, w)
        if s.ratio() >= ratio:
            remove = pair[np.argmin([len(a) for a in pair])]
            remove_list.append(remove)
    remove_list = np.unique(remove_list)
    for entry in remove_list:
        str_list.remove(entry)
    return str_list

def filter_dual_membership(dt):
    for key in dt:
        for sent in dt[key]:
            for kg in dt:
                if kg != key:
                    if bool(np.sum([is_subset(b,sent) for b in dt[kg]])):
                        dt[key].remove(sent)
    return dt

def associate_caption_text(nlp,doc,critical,query_kw=[]):

    if critical != []:
        # Create sequence of spans from caption tokens in 'critical' only
        spans_from_caption, caption_kw = zip(*[(doc[c[2]:c[3]], c[4]) for c in critical])
    else:
        spans_from_caption, caption_kw = (),('0')

    # Get spans from list of punctuations to treated as full-stops
    spans_from_plist = spans_from_list(doc,[";"])

    # Get spans for conjunctions
    spans_from_conj  = spans_from_list(doc,["and"])

    # Get spans for "sticky words" (i.e. NN that often get attached to differentiating JJ such as "high" or "low" by hyphen)
    spans_from_stick = spans_from_list(doc,["magnification"])

    # Get spans for words indicating possible internal "CAP" reference
    spans_from_inref = spans_from_list(doc,["denoted","corresponds","extracted","indicated","shown","corresponding"])

    # Retokenize doc tagging the caption tokens
    doc = custom_retoken(doc,spans_from_caption,"CAP")

    # Retokenize doc tagging semi-colons as full-stops
    doc = custom_retoken(doc,spans_from_plist,".")

    # Retokenize doc tagging the internal ref keywords
    doc = custom_retoken(doc,spans_from_inref,"IR")
    
    # Retokenize doc grouping noun chunks (without overriding prior custom tokens)
    doc = nounchunk_retoken(doc,spans_from_caption+spans_from_plist+spans_from_conj+spans_from_stick+spans_from_inref)

    print("\nCustom tokenize:",[(w.text, w.tag_, w.ent_type_) for w in doc])

    with open("sentence_search_patterns.yml", "r") as stream:
        try:
            token_patterns = [list(ast.literal_eval(a)) for a in yaml.safe_load(stream)['patterns']]
        except yaml.YAMLError as exc:
            print(exc)

    dt, de, dk = {},{},{}

    if critical != []:

        # Caption delimiter entries as keys for token dictionary (dt)
        for key in caption_kw:
            dt.setdefault(key, [])

        # Create list from doc elements --> (text,tag,id)
        tokenized_caption = list(zip([a.text for a in doc],[a.tag_ for a in doc],range(len(doc))))

        # Iterate through tokens to populate dt (populates dictionary based on each caption token key)
        for token_pattern in token_patterns:
            dt = filter_by_tokenized_search_patterns(tokenized_caption,token_pattern,dt)

        # Create by_token list
        for k in list(dt.keys()):
            # Is it unique within?
            selected = cleanup_strings(longest_unrelated_entries(dt[k],0.5),k)
            dt[k] = selected
        # Is unique across caption tokens?
        dt = filter_dual_membership(dt)

        # Create by_explicit list
        for k in list(dt.keys()):
            for key in [a[5] for a in critical if a[4] == k][0]:
                de.setdefault(key, [])
                if de[key] != []:
                    de[key] = de[key]+cleanup_strings(dt[k],k)
                else:
                    de[key] = cleanup_strings(dt[k],k)

        # Create by_keyword list
        for k in list(dt.keys()):
            for key in [a[5] for a in critical if a[4] == k][0]:
                dk.setdefault(key, [])
                if dk[key] != []:
                    selected = cleanup_strings(dt[k],k)
                    if selected != []:
                        selected = associated_dictionary_entry(query_kw,selected[0])
                    dk[key] = list(np.unique(dk[key] + selected))
                else:
                    selected = cleanup_strings(dt[k],k)
                    if selected != []:
                        selected = associated_dictionary_entry(query_kw,selected[0])
                    dk[key] = list(np.unique(selected))
    else:

        dt["(0)"] = doc.text.split(" ")
        de["0"] = doc.text.split(" ")
        dk["0"] = list(np.unique(associated_dictionary_entry(query_kw,doc.text)))

    return dt, de, dk

