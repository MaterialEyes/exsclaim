# -*- coding: utf-8 -*-
import ast
import utils
import collections
import numpy as np

from spacy import load
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler

import captions.interpret as interpret
import captions.regex as regex


def load_models(models_path=str) -> "caption_nlp_model":
    """
    Load a custom spaCy language object + matcher with regex model for full caption parsing 

    Args:
        models_path: A path to folder containing the necessary model files
    
    Returns:
        caption_nlp_model: A tuple (nlp, matcher)

    """
    # Load pre-trained statistical model (English) and input vocab to Matcher
    nlp     = load("en_core_web_sm") 
    matcher = Matcher(nlp.vocab)

    # Get list of caption-spectific spaCy rules (contains label + pattern)
    caption_rules = utils.load_yaml(models_path+'rules.yml')

    # Add custom caption rules to spaCy matcher
    for rule in caption_rules:
        matcher.add(rule['label'],None,rule['pattern'])

    # Add new spans to the Doc.ents and then to the processing pipeline.
    ruler = EntityRuler(nlp, patterns=caption_rules)
    nlp.add_pipe(ruler, before="ner")
    
    return (nlp, matcher) 


def find_subfigure_delimiter(caption_nlp_model=tuple, caption=str) -> str:
    """
    Determine which type of character is functioning as a delimiter 
    between segments of text describing the individual subfigures. 

    Args:
        caption_nlp_model: A tuple (nlp, matcher)
        caption: A string of caption text

    Returns:
        char_delim: The primary delimiter in the list of matches
    """
    nlp, matcher = caption_nlp_model # unpack necessary components of the model
    
    doc     = nlp(caption)
    matches = matcher(doc)

    # Detect delimiters and count frequency of each delimiter type
    delimiters = []
    for m in matches:
        delimiters.append(nlp.vocab.strings[m[0]].split("_")[3])
    c = collections.Counter(delimiters)

    if delimiters == []:
        return None

    # Order priority - delimiter type that comes first
    order_type = delimiters[0]
    
    # Frequency priority and count - delimiter type most common       
    freq_type, freq_count = c.most_common(1)[0] 
    
    # Alpha priority count - max of alpha (lower vs. upper) most common  
    alph  = np.max([c['alpha'],c['ALPHA']]) 
    
    # Alpha priority ratio - ratio of alpha count to count of most frequent delimiter type
    alpha_freq_ratio = float(alph)/freq_count
    
    # Set low b/c often when delimiter is ambiguous, alpha/ALPHA is a suitable decison.
    alpha_thresh=0.25
    
    # If alpha priority ratio high enough, then select the proper alpha/ALPHA. 
    if alpha_freq_ratio >= alpha_thresh:
        char_delim = 'ALPHA' if np.argmax([c['alpha'],c['ALPHA']]) else 'alpha'
    # Elif roman type exists as most frequent, assign it to char_delim.
    elif freq_type == 'roman':
        char_delim = freq_type
    # Otherwise, default char type is the char type that appeared first.
    else:
        char_delim = order_type

    return char_delim


def get_subfigure_tokens(caption_nlp_model=tuple, caption=str) -> list:
    """
    Split caption text into subfigure tokens based on subfigure delimiters. 
    The delimiting subfigure tokens are a given as list of subfigure tokens elements 
    A single subfigure token element is a tuple containing:

        - match_id: The hash value of the string ID 
        - string_id: The the caption rule label -> "rule['label']"
        - start/end: Describes a start/end slice of span in doc, i.e. doc[start:end]
        - doc_text: Text of the span, doc[start:end].text
        - implied_text: Text implied by the doc_text

        i.e (6167809695299526461, 'parenthesis_02_none_alpha_01', 134, 139, '(f–i)', ['f', 'g', 'h', 'i'])
    
    Note: 
        A list of subfigure tokens only refers to portions of the caption text 
        containing a subfigure label. Not all caption text is given a subfigure token!

    Args:
        caption_nlp_model: A tuple (nlp, matcher)
        caption: A string of caption text

    Returns:
        subfigure_tokens:  List of subfigure tokens elements (list of tuples)
    """
    delimiter = find_subfigure_delimiter(caption_nlp_model,caption)

    # Unpack necessary elements of the model
    nlp, matcher = caption_nlp_model 
    doc     = nlp(caption)
    matches = matcher(doc)

    # Find start/stop slices of tokens that contain a delimiter in their 'word_type'
    ss = [list(range(a[1],a[2])) for a in matches \
                 if nlp.vocab.strings[a[0]].split("_")[3] == delimiter]
    ss = [[]]+ss+[[]] # Pad with empty lists to trigger no intersection events at boundaries

    # Find starting points for consecutive slices that do not intersect
    start_idxs = [i for i in range(len(ss)-1) if utils.intersection(ss[i],ss[i+1]) == []]

    # Find idx of max consecutive tokens list between non-intersecting slices
    critical_idxs = list(np.array(start_idxs[0:-1])+\
                         np.array([np.argmax([len(a) \
                            for a in ss[start_idxs[i]+1:start_idxs[i+1]+1]])\
                            for i in range(len(start_idxs)-1)]))

    # Filter out ambiguous tokens (i.e. parenthesis before but not after), and false 
    # positive chemical elements to get collection of "critical" tokens to further inspect
    critical_tokens = [a for a in [matches[i] for i in critical_idxs]\
                         if utils.is_disjoint(doc[a[1]:a[2]].text.split(" "),\
                            interpret.false_negative_subfigure_labels(delimiter))]

    # Find all labels suggested by syntax (i.e. the label a–d suggests that in actuality, images a, b, c, d exist).
    suggested_labels = sorted(np.unique(list(utils.flatten([list(a) \
                   for a in [interpret.implied_chars(doc[entry[1]:entry[2]].text,delimiter)
                   for entry in critical_tokens]]))))
    # At this point, stray "non-chemistry" labels (i.e. "a", "b", "c", "d", "j") 
    # may still be included. Get rid of stray labels if not connected to the group. 
    subfigure_tokens = [] 
    for match_id, start, end in critical_tokens: 
        related = 1 # Assume always related
        if delimiter in ['alpha','ALPHA'] and len(critical_tokens)>1:
            for letter in interpret.implied_chars(doc[start:end].text,delimiter):
                # Cases where not related ...
                if not interpret.is_likely_subfigure(letter,suggested_labels,doc.text):
                   related = 0
        if related:
            implied = interpret.implied_chars(doc[start:end].text,delimiter)
            subfigure_tokens.append((match_id,nlp.vocab.strings[match_id],start,end,doc[start:end].text,implied))

    if subfigure_tokens == []:
        # When entire caption is a single token, assign a blank token.
        subfigure_tokens = [(-99, None, -99, -99, '(0)', ['0'])]

    return subfigure_tokens


def get_subfigure_count(caption_nlp_model=tuple, caption=str) -> int:
    """
    Estimate the number of subfigures from 'implied_text' in subfigure tokens

    Args:
        caption_nlp_model: A tuple (nlp, matcher)
        caption: A string of caption text

    Returns:
        num_subfigs: An integer predicting number of subfigures in a figure
    """
    subfigure_tokens = get_subfigure_tokens(caption_nlp_model,caption)
    
    subfigures = []
    for ct in subfigure_tokens:
        subfigures += ct[5]
    return int(len(np.unique(subfigures)))


def get_subfigure_labels(caption_nlp_model=tuple, caption=str) -> list:
    """
    Get the text for subfigure delimiters from 'doc_text' labels in subfigure tokens

    Args:
        caption_nlp_model: A tuple (nlp, matcher)
        caption: A string of caption text

    Returns:
        subfigure_labels: A list of 'doc_text' labels from subfigure tokens
    """
    subfigure_tokens = get_subfigure_tokens(caption_nlp_model,caption)
    
    subfigure_labels = []
    for cl in subfigure_tokens:
        subfigure_labels.append(cl[4])
    return subfigure_labels


def associate_caption_text(caption_nlp_model=tuple, caption=str, keywords={}, keys='implied') -> "caption_dict":
    """
    Find portions of caption text and assigns them to appropriate subfigure token key 
    in a caption_dict. 

    Args:
        caption_nlp_model: A tuple (nlp, matcher)
        caption: A string of caption text
        keywords: A dictionary of keywords/synonyms pairs that are matched to related caption_dict entries
        keys: A string indicating the style of the 'keys' in the caption_dict output
              "explicit" – caption label keys from the doc_text of the subfigure token: (a–c)
              "implied"  – caption labels keys from the implied_text of a subfigure token : a, b, c.

    Returns:
        caption_dict: A dictionary with subfigure token keys and associated text as entries. 
    """
    # Pre-processing of caption to remove problematic characters/sequences
    caption = interpret.resolve_problem_sequences(caption)

    # Unpack necessary elements of the model
    nlp, __ = caption_nlp_model 
    doc = nlp(caption)
    
    subfigure_tokens = get_subfigure_tokens(caption_nlp_model,caption)
    
    de = regex.caption_sentence_findall(doc, subfigure_tokens, {})

    # Create search dictionary from search_query 'term' keys and 'term'+'synonyms' as entries 
    search_list = {keywords[key]['term']:\
                  [keywords[key]['term']]+keywords[key]['synonyms'] \
                   for key in keywords if len(keywords[key]['term'])>0}

    # Find keywords
    for label in de:
        for keyword in search_list:
            description = " ".join(de[label]['description'])
            if not utils.is_disjoint(search_list[keyword],description.split(" ")):
                de[label]['keywords'].append(keyword)

    if keys == 'explicit':
        return de
    else:
        # Create dictionary with all implied subfigure_label keys and associated text as entries. 
        di = {}
        for token in subfigure_tokens: 
            di.update({k:{"description":de[token[4]]['description'],"keywords":de[token[4]]['keywords']} for k in token[5]})
        return di