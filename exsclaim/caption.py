# -*- coding: utf-8 -*-
import ast
import difflib
import itertools
import collections
import numpy as np

from spacy import load
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler

from .captions import lists
from .utilities import files
from .captions import interpret as interpret
from .captions import regex as regex

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
    caption_rules = files.load_yaml(models_path+'rules.yml')

    # Add custom caption rules to spaCy matcher
    for rule in caption_rules:
        matcher.add(rule['label'], [rule['pattern']])

    # Add new spans to the Doc.ents and then to the processing pipeline.
    ruler = EntityRuler(nlp, name="entity_ruler", patterns=caption_rules)
    nlp.add_pipe("entity_ruler", before="ner")
    
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
        char_delim = 'ALPHA' if np.argmax([1.0*c['alpha'],0.5*c['ALPHA']]) else 'alpha'
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

    # Filter out matches that are not of the correct delimiter
    matches = [a for a in matches \
                 if nlp.vocab.strings[a[0]].split("_")[3] == delimiter]

    # Find start/stop slices of tokens that contain a delimiter in their 'word_type'
    ss = [list(range(a[1],a[2])) for a in matches \
                 if nlp.vocab.strings[a[0]].split("_")[3] == delimiter]
    ss = [[]]+ss+[[]] # Pad with empty lists to trigger no intersection events at boundaries

    # Find starting points for consecutive slices that do not intersect
    start_idxs = [i for i in range(len(ss)-1) if lists.intersection(ss[i],ss[i+1]) == []]

    # Find idx of max consecutive tokens list between non-intersecting slices
    critical_idxs = list(np.array(start_idxs[0:-1])+\
                         np.array([np.argmax([len(a) \
                            for a in ss[start_idxs[i]+1:start_idxs[i+1]+1]])\
                            for i in range(len(start_idxs)-1)]))

    # Filter out ambiguous tokens (i.e. parenthesis before but not after), and false 
    # positive chemical elements to get collection of "critical" tokens to further inspect
    critical_tokens = [a for a in [matches[i] for i in critical_idxs]\
                         if lists.is_disjoint(doc[a[1]:a[2]].text.split(" "),\
                            interpret.false_negative_subfigure_labels(delimiter))]

    # Filter out any remaining tokens that resemble molecules
    critical_tokens = [a for a in critical_tokens if not any(substring in doc[a[1]:a[2]].text for substring in interpret.common_molecules())]


    # Find all labels suggested by syntax (i.e. the label a–d suggests that in actuality, images a, b, c, d exist).
    suggested_labels = sorted(np.unique(list(lists.flatten([list(a) \
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

    if delimiter in ['alpha','ALPHA'] and len(subfigure_tokens)>0:
        A_exist = 0
        for tok in subfigure_tokens:
           if tok[5][0].lower() == "a":
            A_exist = 1

        if not A_exist:
            subfigure_tokens = []

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

    # Find matching caption descriptions
    for label in de:
        # Remove any redundant descriptions when more specific "longer" description is present
        # i.e. reduce ['HAADF-STEM', 'STEM', 'TEM', 'EELS'] to ['HAADF-STEM', 'EELS']
        ratio = 0.2
        remove_list = []
        for pair in itertools.combinations(de[label]['description'], r=2):
            v,w = pair
            s = difflib.SequenceMatcher(None, v, w)
            if s.ratio() >= ratio:
                # If the two strings are similar, add shorter string to remove list
                remove = pair[np.argmin([len(a) for a in pair])]
                remove_list.append(remove)
        remove_list = np.unique(remove_list)
        for entry in remove_list:
            de[label]['description'].remove(entry)

    # Create search dictionary from search_query 'term' keys and 'term'+'synonyms' as entries 
    search_list = {keywords[key]['term']:\
                  [a for a in [keywords[key]['term']]+keywords[key]['synonyms'] if ''!= a] \
                   for key in keywords if len(keywords[key]['term'])>0}

    experiment_list = interpret.experiment_synonyms()

    # Find matching keywords
    for label in de:
        for keyword in search_list:
            description = " ".join(de[label]['description'])
            for slk in search_list[keyword]:
                if len(description.split(slk))>1:
                    if keyword not in de[label]['keywords']:
                        de[label]['keywords'].append(keyword)

    # Find matching general descriptions
    for label in de:
        for keyword in experiment_list:
            description = " ".join(de[label]['description'])
            for elk in experiment_list[keyword]:
                if len(description.split(elk))>1:
                    if keyword not in de[label]['general']:
                        de[label]['general'].append(keyword)

        # Remove any redundant descriptions when more specific "longer" description is present
        # i.e. reduce ['HAADF-STEM', 'STEM', 'TEM', 'EELS'] to ['HAADF-STEM', 'EELS']
        ratio = 0.86
        remove_list = []
        for pair in itertools.combinations(de[label]['general'], r=2):
            v,w = pair
            s = difflib.SequenceMatcher(None, v, w)
            if s.ratio() >= ratio:
                # If the two strings are similar, add shorter string to remove list
                remove = pair[np.argmin([len(a) for a in pair])]
                remove_list.append(remove)
        remove_list = np.unique(remove_list)
        for entry in remove_list:
            de[label]['general'].remove(entry)

    if keys == 'explicit':
        return de
    else:

        # Create dictionary with all implied subfigure_label keys and associated text as entries. 
        implied_keys = list(set([item for sublist in [a[5] for a in subfigure_tokens] for item in sublist]))
        implied_keys.sort()
        di = {}
        
        for k in implied_keys:
            di.update({k:{"description":[],"keywords":[],"general":[]}})

        for token in subfigure_tokens:
            for imptok in token[5]:
                for desc in de[token[4]]['description']:
                    if de[token[4]]['description'] !=[] and desc not in di[imptok]["description"]:
                        di[imptok]["description"].append(desc)

        description_list = []
        for token in subfigure_tokens:
            for imptok in token[5]:
                for entry in di[imptok]["description"]:
                    # Check to see if exact and substring match exists
                    exact_match = True in [True if a.replace(".","").strip() == entry.replace(".","").strip() else False for a in description_list]
                    substring_match = True in [True if a.replace(".","").strip() in entry.replace(".","").strip() else False for a in description_list]

                    # This takes care of case where too much text was retained in the description and the only unique portion of the string is the new information 
                    # This new "unique" information is assigned to the description for the token
                    if not exact_match and substring_match:
                        idx = np.argmax([True if a.replace(".","").strip() in entry.replace(".","").strip() else False for a in description_list])
                        a = description_list[idx]
                        b = entry.replace(".","").strip()
                        unique_text = "".join(b.replace(".","").strip().split(a.replace(".","").strip())).strip()
                        di[imptok]["description"] = [unique_text]
     
                    description_list += di[imptok]["description"]
                    description_list = list(set(description_list))

                for keyw in de[token[4]]['keywords']:
                    if de[token[4]]['keywords'] !=[] and keyw not in di[imptok]["keywords"]:
                        di[imptok]["keywords"].append(keyw)


        # Unclutter strings which are very similar
        for key in di:
            ratio = 0.65
            remove_list = []
            for pair in itertools.combinations(di[key]["description"], r=2):
                v,w = pair
                s = difflib.SequenceMatcher(None, v, w)
                if s.ratio() >= ratio:
                    # If the two strings are similar, add shorter string to remove list
                    remove = pair[np.argmin([len(a) for a in pair])]
                    remove_list.append(remove)
            remove_list = np.unique(remove_list)
            for entry in remove_list:
                di[key]["description"].remove(entry)

        # Find matching general descriptions
        for key in di:
            for keyword in experiment_list:
                description = " ".join(di[key]['description'])
                for elk in experiment_list[keyword]:
                    if len(description.split(elk))>1:
                        if keyword not in di[key]['general']:
                            di[key]['general'].append(keyword)

        # Remove general descriptions that are too similar
        for key in di:
            ratio = 0.6
            remove_list = []
            for pair in itertools.combinations(di[key]["general"], r=2):
                v,w = pair
                s = difflib.SequenceMatcher(None, v, w)
                if s.ratio() >= ratio:
                    # If the two strings are similar, add shorter string to remove list
                    remove = pair[np.argmin([len(a) for a in pair])]
                    remove_list.append(remove)
            remove_list = np.unique(remove_list)
            for entry in remove_list:
                di[key]["general"].remove(entry)

        return di


# THis works and is pretty good, no proximity scoring 
        # # Create dictionary with all implied subfigure_label keys and associated text as entries. 
        # implied_keys = list(set([item for sublist in [a[5] for a in subfigure_tokens] for item in sublist]))
        # implied_keys.sort()
        # di = {}
        
        # for k in implied_keys:
        #     di.update({k:{"description":[],"keywords":[],"general":[]}})

        # for token in subfigure_tokens:
        #     for imptok in token[5]:
        #         for desc in de[token[4]]['description']:
        #             if de[token[4]]['description'] !=[] and desc not in di[imptok]["description"]:
        #                 di[imptok]["description"].append(desc)
        #         # for keyw in de[token[4]]['keywords']:
        #         #     if de[token[4]]['keywords'] !=[] and keyw not in di[imptok]["keywords"]:
        #         #         di[imptok]["keywords"].append(keyw)
        #         # for genr in de[token[4]]['general']:
        #         #     if de[token[4]]['general'] !=[] and genr not in di[imptok]["general"]:
        #         #         di[imptok]["general"].append(genr)

        # description_list = []
        # for token in subfigure_tokens:
        #     for imptok in token[5]:

        #         for entry in di[imptok]["description"]:
        #             # Check to see if exact and substring match exists
        #             exact_match = True in [True if a.replace(".","").strip() == entry.replace(".","").strip() else False for a in description_list]
        #             substring_match = True in [True if a.replace(".","").strip() in entry.replace(".","").strip() else False for a in description_list]

        #             # This takes care of case where too much text was retained in the description and the only unique portion of the string is the new information 
        #             # This new "unique" information is assigned to the description for the token
        #             if not exact_match and substring_match:
        #                 idx = np.argmax([True if a.replace(".","").strip() in entry.replace(".","").strip() else False for a in description_list])
        #                 a = description_list[idx]
        #                 b = entry.replace(".","").strip()
        #                 unique_text = "".join(b.replace(".","").strip().split(a.replace(".","").strip())).strip()
        #                 di[imptok]["description"] = [unique_text]
     
        #             description_list += di[imptok]["description"]
        #             description_list = list(set(description_list))

        #         for keyw in de[token[4]]['keywords']:
        #             if de[token[4]]['keywords'] !=[] and keyw not in di[imptok]["keywords"]:
        #                 di[imptok]["keywords"].append(keyw)
        #         for genr in de[token[4]]['general']:
        #             if de[token[4]]['general'] !=[] and genr not in di[imptok]["general"]:
        #                 di[imptok]["general"].append(genr)


        # # Unclutter strings which are very similar
        # for key in di:
        #     ratio = 0.33
        #     remove_list = []
        #     for pair in itertools.combinations(di[key]["description"], r=2):
        #         v,w = pair
        #         s = difflib.SequenceMatcher(None, v, w)
        #         if s.ratio() >= ratio:
        #             # If the two strings are similar, add shorter string to remove list
        #             remove = pair[np.argmin([len(a) for a in pair])]
        #             remove_list.append(remove)
        #     remove_list = np.unique(remove_list)
        #     for entry in remove_list:
        #         di[key]["description"].remove(entry)



        # print(di)
        # return di