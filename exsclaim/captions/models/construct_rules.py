# -*- coding: utf-8 -*-
import yaml

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

def caption_nlp_model():
    """
    Add custom caption components to spaCy nlp model 

    # Description of parameters used to create caption pattern collection:
    :param: offsets: punctuation used to set off characters that are explanatory (i.e. a parenthesis)
    :param: position_keys : position of the offset character: 0-before, 1-after, 2-both
    :param: separations: delimiter between characters within the offsets
    :param: char_types: the character type (letter-> alpha or ALPHA (capitalized), digit-> number ... etc)
    :param: char_nums: number of consecutive characters between delimeters
    :param: custom_pattern: a list of tuples containing any custom patterns (from observation)

    :return: v
    """

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

    with open('rules.yml', 'w') as yaml_file:
        yaml.dump(caption_patterns, yaml_file)

if __name__== "__main__":
  caption_nlp_model()  
