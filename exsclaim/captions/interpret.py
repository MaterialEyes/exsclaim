import os
import re
import yaml
import itertools
import numpy as np
from itertools import chain

def load_chrz():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/models/characterization.yml', 'r') as stream:
        try:
            chrz = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return chrz

def load_ref():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/models/reference.yml', 'r') as stream:
        try:
            ref = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return ref

def experiment_synonyms():
    chrz = load_chrz()
    return chrz

def greek_letters():
    greek_codes = chain(range(0x370,0x3e2),range(0x3f0,0x400))
    greek_symbols = (chr(c) for c in greek_codes)
    return [c for c in greek_symbols if c.isalpha()]

def implied_chars(str_text: str, char_type: str) -> str:
    """
    Find the all chars implied by string syntax.

    Example:
        >>> interpret.implied_chars('a and d','alpha')
            ['a', 'd']
        >>> interpret.implied_chars('a–d','alpha')
            ['a', 'b', 'c', 'd']

    Args:
        str_text: A string of text
        char_type: The character type (alpha, digit, or roman)

    Returns:
        implied_char_list: A list containing the chars implied by string syntax
    """
    ref = load_ref()
    str_text = (str_text.strip("(").strip(")").strip(" "))
    str_text = str_text.strip(".")
    str_text = str_text.replace("and"," , ").replace(" ","").replace("-","–").replace(":","")

    discrete_tokens = [","]
    continuous_tokens = ["–"]
    gl = greek_letters()

    token_list = discrete_tokens + continuous_tokens
    token_key = ("|").join(token_list)

    chars = re.split(token_key,str_text)
    delims = [a for a in str_text if a in token_list]

    # Interleave chars with delims
    str_text_list = [x for x in itertools.chain.from_iterable(itertools.zip_longest(chars,delims)) if x is not None]
    str_text_list = [''.join([i for i in a if not i.isdigit()]) for a in str_text_list]

    if char_type == 'position':
        return [a for a in str_text_list if a.lower() in ref["positions"]]

    if len(str_text_list) == 1:
        return str_text_list[0].split(",")

    IC = ""
    for i in np.arange(0,len(str_text_list)-1,2):
        if str_text_list[i+1] in discrete_tokens:
            DT = ",".join([str_text_list[i],str_text_list[i+2]])
        else:
            DT = ""
        if (str_text_list[i+1] in continuous_tokens) and (str_text_list[i] not in gl) and (str_text_list[i+2] not in gl):
            CT = ",".join(char_range(str_text_list[i],str_text_list[i+2],char_type))
        else:
            CT = ""
        IC += DT+CT+","
    joined = ",".join([a for a in np.unique(IC[:-1].split(",")) if len(a)>0])
    return joined.replace(":","").replace("(","").replace(")","").split(",")


def char_range(start: str, stop: str, char_type: str) -> str:
    """
    Return evenly spaced values within a given interval provided with "char_type" context.

    Example:
        >>> interpret.char_range('iv','x','roman')
            ['iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
        >>> interpret.char_range('a','e','alpha')
            ['a', 'b', 'c', 'd', 'e']
    Args:
        start : Starting letter, digit, or roman numeral (string or int)
        stop: End of interval. The interval does include this value.
        char_type: The character type (alpha, digit, or roman)

    Returns:
        char_list: A list containing evenly spaced values within a given interval.
    """
    ref = load_ref()
    if "" == start or "" == stop:
        return [""]

    # Prevent multicharacter non-numeric start/stop points
    if not (len(start)<3 or start.isnumeric()) or not (len(stop)<3 or stop.isnumeric()):
        return [""]

    # Prevent roman from being mislabeled as "alpha" 
    if len(start)+len(stop)>2:
        char_type = 'roman'

    if (len(start) > 1 or len(stop) > 1):
        if start not in ref['roman numerals'] and stop not in ref['roman numerals']:
            return [""]
        else:
            pass

    if char_type.lower() == 'alpha' or char_type == 'ALPHA':
        try:       
            st = ref['alphabet'].index(start.lower())
            ed = ref['alphabet'].index(stop.lower())

            if start == start.upper():
                return [a.upper() for a in ref['alphabet'][st:ed+1]]
            else:
                return [a for a in ref['alphabet'][st:ed+1]]
        except:
            return 

    elif char_type.lower() == 'roman':
        try: 
            st = ref['roman numerals'].index(start.upper())
            ed = ref['roman numerals'].index(stop.upper())

            if start == start.upper():
                return [a for a in ref['roman numerals'][st:ed+1]]
            else:
                return [a.lower() for a in ref['roman numerals'][st:ed+1]]
        except:
            return 

    elif char_type.lower() == 'digit':
        return [str(a) for a in np.arange(start,stop+1)]


def is_likely_subfigure(label: str, suggested_labels: list, caption_text: str) -> bool:
    """
    Determine if the label (alpha only) is a probable subfigure label implied from caption text.
    The purpose of this function is to overwrite likely "false negative" label deletions. 
    "Probable" means one (or more) of the folowing: 
        (1) label is contained in the span of suggested_labels
        (2) label is attached via a connections list
        (3) label is immediately adjacent to a suggested_label
    
    Example:
        interpret.is_probable_subfigure('a',['b','c','d'],\
        "(a) Cross-sectional BSE image of the crystal grain including CaSi2FX compound. (b) EPMA quantitative...")
        >> True
        NOTE: 'a' is likely a subfigure label despite fact it was deleted from suggested_labels!

    Args:
        label: A single alpha char 
        label: A list of estimated subfigure labels
        caption_text: A string of caption text

    Returns:
        A decision on whether the label is likely a true subfigure label
    """
    ref = load_ref()
    # Ensure the label to compare is of "alpha/ALPHA" type
    try:
        label_id = ref['alphabet'].index(label.lower())
    except:
        return False

    # Order the suggested_labels list and remove "label" from it
    suggested_labels = sorted(list(set([a for a in suggested_labels if a.lower() in ref['alphabet'] \
                                  and a.lower()!=label.lower()])))
    
    # Get index of final entry in suggested_labels list
    try:
        final_id = ref['alphabet'].index(suggested_labels[-1].lower())
    except:
        final_id = -1

    # Find all segments of label enclosed in parenthesis and groups each segment into "connections",
    # a list of lists where each list represents a group label chars related by parenthesis
    # (i.e. a–d implies that "d" is connected to "a")

    connections = []

    string_subs = [[")–(","–"],[")-(","–"]] # Resolve known problematic string sequences
    for subs in string_subs:
        caption_text = " ".join([a.replace(subs[0],subs[1]) for a in caption_text.split(" ")])

    caption_text = re.findall('\(.*?\)',caption_text)
    
    for entry in caption_text:
        elem = str(re.sub(r'[^\w\s]',' ',entry)).split(" ")
        connections.append([a for a in elem if a not in ['and','']])

    isconnected = 0
    for entry in connections:
        if label in entry:
            if len([value for value in entry if value in suggested_labels])>0:
                isconnected = 1

    isnumeric = 0
    for entry in connections:
        if label in entry:
            if len([s for s in entry if s.isdigit()]) > 0:
                isnumeric = 1

    isadjacent = 0
    for entry in suggested_labels:
        if np.abs(label_id-ref['alphabet'].index(entry.lower())) == 1:
            isadjacent = 1

    # If label is contained within bounds of suggested_labels, OR if its connected, OR adjacent = True!
    if final_id > label_id or (isconnected and not isnumeric) or isadjacent:
        return True
    else:
        return False


def resolve_problem_sequences(str_text: str) -> str:
    """
    Resolves known problematic string sequences (All user-defined from observation)
    
    NOTE: 
        This function is intended to be further improved 
        with more examples and further testing. 

    Args:
        str_text: A string of text

    Returns: 
        str_text: Plain string UTF decoded with problem sequences resolved
    """
    def resolve_semicolon_abuse(str_text: str, trigger_pair: list) -> str:
        """
        Finds text between trigger_pair words, where authors separated simple dependent 
        clauses with semi-colons, and replace the affected semi-colons with commas. 

        Example:
            If semi-colons were used incorrectly, replace with commas between 
            text in the trigger pair.

                I need the weather statistics for the following 
                cities: London; Paris; Perth; and Brussels.
            
            $ resolve_semicolon_dependent_clause(str_text,["cities"," ."])

                I need the weather statistics for the following 
                cities: London, Paris, Perth, and Brussels.

        Args:
            str_text: A string of text

        Returns: 
            A resolved string of text
        """
        # Identifies the primary char used as full-stop for sentence delimiting.
        # NOTE: This is a naive frequency-based approach without NLP.
        fs = ("; ",". ")[str_text.count(". ")>=str_text.count("; ")]

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

    # Ensures UTF-8 can be used in strings
    if type(str_text) != str:
        str_text =  str_text.decode('utf-8')

    # Resolve known problematic string sequences
    string_subs = [[")-(","-"],[")–(","–"],[":(",": ("],["):",") :"]]
    for subs in string_subs:
        str_text = " ".join([a.replace(subs[0],subs[1]) for a in str_text.split(" ")])

    # Enforce single space
    str_text = re.sub(' +', ' ',str_text)
    
    # Attempt to correct common/observed grammar mistakes with semicolons
    str_text = resolve_semicolon_abuse(str_text,["images of"," ."])
    str_text = resolve_semicolon_abuse(str_text,["with precipitates"," ."])

    # Custom substitution of redundant phrases to assist in caption correspondence. 
    str_text = str_text.replace(". From left to right: (a)",": (a)")
    str_text = str_text.replace("of image (","shown in image (")
    # str_text = str_text.replace(") and (",",")
    str_text = str_text.replace("(a) :",": (a)")
    str_text = str_text.replace("1 –","1 -")
    str_text = str_text.replace("2 –","2 -")
    str_text = str_text.replace("3 –","3 -")
    str_text = str_text.replace("N-C.","N-C .")
    str_text = str_text.replace(". "," . ")
    # str_text = str_text.replace("a. ","a . ")
    # str_text = str_text.replace("b. ","b . ")
    # str_text = str_text.replace("c. ","c . ")
    # str_text = str_text.replace("d. ","d . ")
    # str_text = str_text.replace("e. ","e . ")
    # str_text = str_text.replace("f. ","f . ")
    # str_text = str_text.replace("g. ","g . ")
    str_text = str_text.replace(". (a)"," (a)")

    try:
        if(str_text[0] == 'a' and str_text[1] == " "):
            str_text = "(a)"+str_text[1:]
        elif(str_text[0] == 'A' and str_text[1] == " "):
            str_text = "(A)"+str_text[1:]
    except:
        pass

    try:
        if(str_text[-1] != "."):
            str_text = str_text+" ."
    except:
        pass
    
    return str_text


def false_negative_subfigure_labels(char_delim: str) -> list:
    """
    Create list of false negative subfigure labels based on delimiter
    
    Args:
        char_delim: The primary subfigure delimiter in the caption

    Returns: 
        fp_labels: List of possible FN labels
    """
    ref = load_ref()
    if char_delim != 'ALPHA':
        elements = [a for a in ref['chemical elements']] + \
                   [a.upper() for a in ref['chemical elements']]
    else:
        elements = [a for a in ref['chemical elements'] if len(a) > 1] + \
                   [a.upper() for a in ref['chemical elements'] if len(a) > 1]
    ambiguous_tokens  = ["("+a for a in ref["alphabet"]]
    fn_labels = elements+ambiguous_tokens
    return fn_labels

def common_molecules() -> list:
    """
    """
    ref = load_ref()
    return [a for a in ref["common molecules"]]