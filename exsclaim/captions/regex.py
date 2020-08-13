import os
import ast
import yaml
import difflib
import itertools
import numpy as np

def load_caption_sentence_regex():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/models/patterns.yml', 'r') as stream:
        try:
            caption_sentence_regex = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return caption_sentence_regex

def get_caption_tokenization(doc="spacy.tokens.doc.Doc", subfigure_tokens=list) -> "spacy.tokens.doc.Doc":
    """
    Make captions-specific modifications to the Doc’s tokenization 

    Args:
        doc: A default spaCy tokenized doc (spacy.tokens.doc.Doc)
        subfigure_tokens: A list of subfigure tokens elements (list of tuples)

    Returns:
        doc: A retokenized doc (spacy.tokens.doc.Doc)
    """

    def filter_spans(spans):
        """
        Filter a sequence of spans so they don't contain overlaps!
        """
        get_sort_key = lambda span: (span.end - span.start, span.start)
        sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
        result = []
        seen_tokens = set()
        for span in spans:
            if span.start not in seen_tokens and span.end-1 not in seen_tokens:
                result.append(span)
                seen_tokens.update(range(span.start, span.end))
        return result
    
    def spans_from_list(doc="spacy.tokens.doc.Doc", keywords=list) -> "spacy.tokens.span.Spans":
        """
        Get the keyword spans contained in doc

        Args:
            doc: A spaCy processed document (spacy.tokens.doc.Doc)
            keywords: List of keywords to tokenize

        Returns:
            spans: tuple containing series of (spacy.tokens.span.Spans)
        """
        return tuple([doc[i:i+1] for i in range(len(doc)) if doc[i:i+1].text in keywords])

    def custom_retoken(doc="spacy.tokens.doc.Doc", custom_spans="spacy.tokens.span.Spans", custom_tag=str) -> "spacy.tokens.doc.Doc":
        """
        Make modifications to the Doc’s tokenization following custom_spans and custom_tag input

        Args:
            doc: A spaCy processed document (spacy.tokens.doc.Doc)
            custom_spans: tuple containing series of (spacy.tokens.span.Spans)
            custom_tag: tag "str" assigned to custom_spans 

        Returns:
            doc: A spaCy processed document (spacy.tokens.doc.Doc)
        """
        custom_spans = filter_spans(custom_spans)
        with doc.retokenize() as retokenizer:
            for span in custom_spans:
                retokenizer.merge(span,attrs={"TAG":custom_tag})
        return doc

    def nounchunk_retoken(doc="spacy.tokens.doc.Doc", remove_spans="spacy.tokens.span.Spans") -> "spacy.tokens.doc.Doc":
        """
        Make modifications to the Doc’s tokenization following custom_spans and custom_tag input

        Args:
            doc: A spaCy processed document (spacy.tokens.doc.Doc)
            remove_spans: tuple containing series of spans to remove (spacy.tokens.span.Spans)

        Returns:
            doc: A spaCy processed document (spacy.tokens.doc.Doc)
        """
        # Combine tokens to form noun chunks, and remove already defined tokens
        noun_chunk_spans = []
        for chunk in doc.noun_chunks:
            spans_to_remove = [chunk[i:i+1] for i in range(len(chunk)) if chunk[i:i+1] in remove_spans]
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
                try:
                    retokenizer.merge(span,attrs={"TAG":"NC"})
                except:
                    pass
        return doc

    if subfigure_tokens != [(-99, None, -99, -99, '(0)', ['0'])]:
        # Create sequence of spans from subfigure tokens
        ctok = [(doc[subfigure_tokens[0][2]:subfigure_tokens[0][3]],subfigure_tokens[0][4])]
        for i in range(1,len(subfigure_tokens)):
            if subfigure_tokens[i][3] > subfigure_tokens[i-1][3]: # To avoid merging non-disjoint spans.
                ctok.append((doc[subfigure_tokens[i][2]:subfigure_tokens[i][3]], subfigure_tokens[i][4]))
        spans_from_caption, caption_kw = zip(*ctok)
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

    # Retokenize doc tagging the subfigure tokens
    doc = custom_retoken(doc,spans_from_caption,"CAP")

    # Retokenize doc tagging semi-colons as full-stops
    doc = custom_retoken(doc,spans_from_plist,".")

    # Retokenize doc tagging the internal ref keywords
    doc = custom_retoken(doc,spans_from_inref,"IR")
    
    # Retokenize doc grouping noun chunks (without overriding prior custom tokens)
    doc = nounchunk_retoken(doc,spans_from_caption+spans_from_plist+spans_from_conj+spans_from_stick+spans_from_inref)

    # print("\nCustom tokenize:",[(w.text, w.tag_, w.ent_type_) for w in doc])
    return doc


def get_caption_chunks(doc="spacy.tokens.doc.Doc", subfigure_tokens=list) -> list:
    """
    Return full list of caption chunk elements. A caption chunk element 
    is a groups of words that are paired together during tokenization and 
    are represented as a tuple containing:

        - doc_text : (str) The original word (group of words) text
        - tag (POS): (str) The detailed part-of-speech tag (custom or normal).
        - index    : (int) Order/position in tokenized caption text

        i.e. ('(a–c)', 'CAP', 0) or ('SEM image', 'NC', 1), etc ...
    
    Note: 
        A list of caption chunks encapsulates the full text of the the caption,
        not just the subfigure delimiters!

    Args:
        caption_nlp_model: A tuple (nlp, matcher, \
                           caption_sentence_regex, caption_reference)
        caption: A string of caption text

    Returns:
        caption_chunks: A list of caption_chunk elements (list of tuples)
    """
    doc = get_caption_tokenization(doc, subfigure_tokens)
    return list(zip([a.text for a in doc],[a.tag_ for a in doc],range(len(doc))))


def caption_sentence_search(caption_chunks=list, pattern=str, caption_dict=dict) -> "caption_dict":
    """
    Find sentence segments that conform to sentence regex

    Args:
        caption_chunks: A list of caption_chunk elements (list of tuples)
        pattern : A single caption_sentence_regex pattern
        caption_dict: A dictionary with subfigure tokens (doc_text) keys and associated text as entries. 

    Returns:
        caption_dict: A dictionary with subfigure tokens (doc_text) keys and associated text as entries. 
    """
    def group_consecutive(caption_chunks=list) -> list:
        """
        Groups neighboring list entries into single list. List entries are considered 
        neighbors as long as they do not border "*" or "!"

        Args:
            caption_chunk_list: List of caption chunks

        Returns: 
            grouped_list: A list of lists where neighbors are grouped into a list (segment)
        """
        grouped_list = []
        sub = []
        for a in caption_chunks:
            if a[0] != "*" and a[0] != "!":
                sub.append(a)
            else:
                grouped_list.append(sub)
                grouped_list.append([(a,a)])
                sub = []
        if sub != []:
            grouped_list.append(sub)
        return grouped_list

    def segment_position(segment=list, full_list=list) -> tuple:
        """
        Find position (start, end) of a segment of POS tags in a full POS tag list.

        Args:
            segment: A partial list of POS_tag's
            full_list: A full list of POS_tag's

        Returns: 
            position: A tuple giving start, end of the position of the partial list in the full list.
        """
        i = 0
        while True:
            try:
                i = segment.index(full_list[0], i)
            except ValueError:
                return (-1,-1)
            if full_list == segment[i:i+len(full_list)]:
                return (i,i+len(full_list))
            i += 1

    def fill_gaps(tuple_list):
        """
        Fill gaps between non-consecutively indexed tuples in tuple_list

        Args:
            tuple_list: A list of tuples with (idx,0/1)

        Example:
            [(1, 1), (2, 1), (5, 1)] becomes [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]

        Returns: 
            tuple_list: A tuple_list with gaps between previously non-consecutively indexed tuples filled
        """
        lc = tuple_list.copy()
        cnt = 0
        for a,b in tuple_list:
            if a > cnt:
                tuple_list.insert(cnt, tuple((cnt, 1)))
            cnt += 1
        return tuple_list[tuple_list.index(lc[0])::]

    # Groups neighboring list entries (* and ! define neighbor borders) into single list.
    grouped_list = group_consecutive(pattern)

    # Iteratively move the anchor of the search 
    for anchor in range(len(caption_chunks)):
        # Get current start/end point
        cc_beg, cc_end, intervals = anchor, len(caption_chunks), []
        # Search for first segment of pattern in the caption
        for segment in grouped_list: 
            # Remaining caption sequence with the current starting point
            remaining_caption = caption_chunks[cc_beg:cc_end:1]
            # Get all the POS tokens in the token_pattern subsequence
            segment_pos = [a[0] for a in segment]
            # Get all the POS tokens in the caption sequence
            caption_pos =  [a[1] for a in remaining_caption]
            # If not an "include/exclude until" wildcard
            if segment_pos[0] != "*" and segment_pos[0] != "!":
                # Get start/end points for existence of the segment pos in the caption pos
                seg_beg,seg_end = segment_position(caption_pos, segment_pos)
                if seg_beg == -1: # Subsequence does not exist
                    break
                else:
                    # Resume search next time through remaining_caption at end of the critical slice
                    cc_beg = remaining_caption[seg_end-1][2]+1 
                    # Get the absolute indices of the critical slice in remaining_caption
                    critical_idxs = [a[2] for a in remaining_caption[seg_beg:seg_end]]
                    # Create list of tuples for critical slice.
                    # Tuples contain absolute index and T/F inclusion flag (i.e. [(120, 1), (121, 1)])             
                    idxs_switch = list(zip(critical_idxs,[a[1] for a in segment]))
                    intervals.extend(idxs_switch)
            else:
                # If wildcard/stop key is a stop, append to current list to indicate break
                if segment_pos[0] == "!":
                    intervals.extend(segment_pos) # Token to indicate break!
        # All segment in grouped_list must exist (i.e no seg_beg = -1) to continue
        if seg_beg > -1:
            # Group contiguous intervals and fill implied gaps 
            if intervals != []:
                intervals_extended = []
                for gl in group_consecutive(intervals):
                    if gl[0] != ('!', '!') and len(gl)>0:
                        intervals_extended.extend(fill_gaps(gl)) 
            # In an effort to eliminate solutions with multple "CAP"s and with full-stops in the middle...
            cap_count = [caption_chunks[a[0]][1] for a in intervals_extended if a[1] == 1].count('CAP')
            
            # (*) Method #1 (lenient)- only search for stops among characters that are not ignored!
            midfs_count = [caption_chunks[a[0]][1] for a in intervals_extended[0:-1]].count('.')
            issue_1 = midfs_count > 3

            # # (*) Method #2 (strict) -  search for stops occuring between endpoints
            # pos_list = [caption_chunks[a][1] for a in list(np.arange(intervals_extended[0][0],intervals_extended[-1][0]))]
            # midfs_count = pos_list.count('.') + pos_list.count(';')
            # issue_1 = midfs_count > 0

            # Count number of internal references made (cue to include additional CAP separate from the remaining_caption CAP)
            # i.e. we want to keep (b) in this situation: (c) HAADF STEM image of the the region indicated in (b). 
            iref_count  = [caption_chunks[a[0]][1] for a in intervals_extended if a[1] == 1].count('IR')
            # Retain only sentences with CAPS after the IR (there are instances where this fails, but should fail conservatively)
            iref_valid = True
            if iref_count:
                reversed_pos_list = [caption_chunks[a[0]][1] for a in reversed(intervals_extended) if a[1] == 1]
                if reversed_pos_list.index("IR") < reversed_pos_list.index("CAP"):
                    iref_valid = False
            # Issue #1:
            # Full stop punctuation in the middle of the line.
            # issue_1 = midfs_count > 3
            # issue_1 = midfs_count > 0
            # Issue #2:
            # Multiple CAP refs without an IR indicator, 
            issue_2 = (cap_count >1 and iref_count==0)
            # Issue #3:
            # More than 2 CAP refs for a single IR indicator
            issue_3 = (cap_count > 2 and iref_count==1)
            # Issue #4:
            # Multiple CAP refs with an IR indicator, but invalid IR
            issue_4 = (cap_count >1 and iref_count==1 and iref_valid==False)
            # If none of the issues are a problem, add to caption_dict
            if not (issue_1 or issue_2 or issue_3 or issue_4):
                # Assign text string to governing CAP
                cap_str = [caption_chunks[a[0]][0] for a in intervals_extended if a[1]==1 and caption_chunks[a[0]][1]=='CAP'][0]
                # Convert sequence of match tokens to text string
                match = " ".join([caption_chunks[a[0]][0] for a in intervals_extended if a[1]==1])
                # Assign key (determined by governing CAP) to the matched text string. 
                caption_dict[cap_str]['description'].append(match.replace("( ","(").replace(" )",")"))
    
    # Retain only unique values (entries)
    for k in list(caption_dict.keys()):
        caption_dict[k]['description'] = list(np.unique(caption_dict[k]['description']))
    
    return caption_dict


def consolidate_entries(caption_dict=dict)-> "caption_dict":
    """
    Iterate through dict keys and consolidate strings in each entry list based on 
    similarity and relevance

    Args:
        caption_dict: A dictionary with subfigure tokens (doc_text) keys and associated text as entries. 

    Returns:
        caption_dict: A dictionary with subfigure tokens (doc_text) keys and associated (consolidated) text as entries. 
    """
    def filter_dual_membership(caption_dict=dict) -> dict:
        """
        Iterate through dict keys in order and if an entry not containing a key's subfigure label 
        is assigned to another key, delete it from the current key.

        Args:
            caption_dict: A dictionary with subfigure tokens (doc_text) keys and associated text as entries. 

        Returns:
            caption_dict: A dictionary with subfigure tokens (doc_text) keys and associated (filtered) text as entries. 
        """
        for key in caption_dict:
            remove_list = []
            for sent in caption_dict[key]['description']:
                for kg in caption_dict:
                    if kg != key:
                        if bool(np.sum([is_subset(b,sent) for b in caption_dict[kg]['description']])):
                            remove_list.append(sent)
            for rem in np.unique(remove_list):
                caption_dict[key]['description'].remove(rem)
        return caption_dict

    def is_caption_token_mismatch(str_1=str, str_2=str) -> bool:
        """
        See if string mismatch occurs because of initial caption token

        Args:
            str_1: string 1
            str_2: string 2

        Results:
            boolean
        """
        s = difflib.SequenceMatcher(None, str_1, str_2)
        match_elems = []
        for block in s.get_matching_blocks():
            start_match, _, _ = block
            if start_match == 0:
                return False
            else:
                return True
    
    def is_subset(str_1=str, str_2=str, tol=0)-> bool:
        """
        See if str_2 is a subset of str_1 
        (i.e. portions of str_2 exist in str_2)

        Args:
            str_1: string 1
            str_2: string 2

        Results:
            boolean
        """
        s = difflib.SequenceMatcher(None, str_1, str_2)
        match_elems = []
        for block in s.get_matching_blocks():
            _ , _ , match = block 
            match_elems.append(match)
        return np.max(match_elems) >= len(str_2)-np.floor((tol*len(str_2)))

    for key in caption_dict:
        # Keep longest unrelated entries in list
        str_list = list(np.unique(caption_dict[key]['description']))

        ratio = 0.7
        remove_list = []

        # Go through all entries and remove all that contain ". a" style subfigure labels
        # for entry in str_list:
        #     if key in entry:
        #         if entry.find(key)>1:
        #             remove_list.append(entry)

        # Go through all pairs of strings in list and calculate similarity ratio
        for pair in itertools.combinations(str_list, r=2):
            v,w = pair
            s = difflib.SequenceMatcher(None, v, w)
            if s.ratio() >= ratio:
                # If the two strings are similar, add shorter string to remove list
                remove = pair[np.argmin([len(a) for a in pair])]
                remove_list.append(remove)
        remove_list = np.unique(remove_list)
        for entry in remove_list:
            str_list.remove(entry)
        # Remove key from string only if it is at the beginning
        str_list_notok = [a.replace(key,"",1) for a in str_list]
        str_resolved = []
        for i in range(len(str_list)):
            if is_caption_token_mismatch(str_list[i],str_list_notok[i]):
                str_resolved.append(str_list_notok[i])
            else:
                str_resolved.append(str_list[i])
        # Remove superfluous spaces and punctuation marks
        selected = [text.strip(",").strip(".").strip(";").strip(" ").replace(" ,",",")+"." for text in str_resolved]
        caption_dict[key]['description'] = selected

    # Is unique across caption tokens?
    return filter_dual_membership(caption_dict)


def caption_sentence_findall(doc="spacy.tokens.doc.Doc", subfigure_tokens=list, caption_dict=dict) -> "caption_dict":
    """
    Find portion of caption text corresponding to each subfigure key 
    using a sentence-level regex from dictionary of sequences of "ON/OFF" POS tags 
    and "include/exclude until" wildcard patterns in "patterns.yml"

    Args:
        caption_nlp_model: A tuple (nlp, matcher, \
                           caption_sentence_regex, caption_reference)
        caption: A string of caption text
        caption_dict: A dictionary with subfigure tokens (doc_text) keys and associated text as entries. 

    Returns:
        caption_dict: A dictionary with subfigure tokens (doc_text) keys and associated text as entries. 
    """
    caption_sentence_regex = load_caption_sentence_regex()
    caption_chunks = get_caption_chunks(doc, subfigure_tokens)

    # View caption chunks:
    # print("\nCAPTION CHUNKS: ",caption_chunks)

    subfigure_labels = []
    for cl in subfigure_tokens:
        subfigure_labels.append(cl[4])

    # Create dictionary with explicit subfigure_label keys and associated text as entries. 
    if caption_dict == {}:
        caption_dict = {k:{"description":[],"keywords":[],"general":[]} for k in subfigure_labels}
    
    if "(0)" in caption_dict:
        caption_dict["(0)"]['description'] = " ".join([a[0] for a in caption_chunks])
    else: 
        for sentence_pattern in caption_sentence_regex['patterns']:
            caption_dict = caption_sentence_search(caption_chunks,list(ast.literal_eval(sentence_pattern)),caption_dict)
    return consolidate_entries(caption_dict)