import sys
import ast
import yaml
import numpy as np
from captions.tools.helper_nlp import *

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def associate_caption_text(nlp,doc,critical,query_kw=[],tokens_path=''):

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

    with open(tokens_path, "r") as stream:
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
        dt["(0)"] = [str(doc.text)]
        de["0"] = [str(doc.text)]
        dk["0"] = list(np.unique(associated_dictionary_entry(query_kw,doc.text)))

    return dt, de, dk

def parse_caption(caption,keywords,nlp,matcher,tokens_path,query_kw=""):
    """
    Matches sequences of caption-relevant tokens to caption text for purposes of segmenting 
    the caption text into segments corresponding to an individual subfigure images. 

    :return: spaCy matcher object (with explicit connections between implied subfigures and tokens), Predicted number of images (subfigures)
    """
    blockPrint()
    
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
    # query_kw = {"HAADF-STEM":["HAADF","HAADF-STEM","HAADF–STEM","High-angle annular dark-field","High angle annular dark-field","High-angle annular darkfield","High angle annular darkfield"]}
                # "Z-contrast":["Z-contrast","Z–contrast"]}

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

    dt, de, dk = associate_caption_text(nlp,doc,resolved,query_kw,tokens_path)

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
    enablePrint()
    return num_imgs_implied, dt, de, dk 