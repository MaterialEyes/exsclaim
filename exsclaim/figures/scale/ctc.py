#### ADAPTED FROM https://github.com/githubharald/CTCDecoder/blob/master/src/BeamSearch.py
from __future__ import division
from __future__ import print_function
import numpy as np
import pathlib
from .lm import LanguageModel
from operator import itemgetter


class BeamEntry:
    "information about one single beam at specific time-step"
    def __init__(self):
        self.prTotal = 0 # blank and non-blank
        self.prNonBlank = 0 # non-blank
        self.prBlank = 0 # blank
        self.prText = 1 # LM score
        self.lmApplied = False # flag if LM was already applied to this beam
        self.labeling = () # beam-labeling


class BeamState:
    "information about the beams at specific time-step"
    def __init__(self):
        self.entries = {}

    def norm(self):
        "length-normalise LM score"
        for (k, _) in self.entries.items():
            labelingLen = len(self.entries[k].labeling)
            self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

    def sort(self):
        "return beam-labelings, sorted by probability"
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)
        return [(x.labeling, x.prTotal*x.prText) for x in sortedBeams]


def applyLM(parentBeam, childBeam, classes, lm):
    "calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars"
    if lm and not childBeam.lmApplied:
        c1 = classes[parentBeam.labeling[-1] if parentBeam.labeling else classes.index(' ')] # first char
        c2 = classes[childBeam.labeling[-1]] # second char
        lmFactor = 0.01 # influence of language model
        bigramProb = lm.getCharBigram(c1, c2) ** lmFactor # probability of seeing first and second char next to each other
        childBeam.prText = parentBeam.prText * bigramProb # probability of char sequence
        childBeam.lmApplied = True # only apply LM once per beam entry


def addBeam(beamState, labeling):
    "add beam if it does not yet exist"
    if labeling not in beamState.entries:
        beamState.entries[labeling] = BeamEntry()


def ctcBeamSearch(mat, classes, lm, beamWidth=25):
    "beam search as described by the paper of Hwang et al. and the paper of Graves et al."

    blankIdx = len(classes)
    maxT, maxC = mat.shape

    # initialise beam state
    last = BeamState()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].prBlank = 1
    last.entries[labeling].prTotal = 1

    # go over all time-steps
    for t in range(maxT):
        curr = BeamState()

        # get beam-labelings of best beams
        bestLabelings = last.sort()[0:beamWidth]

        # go over best beams
        for labeling, conf in bestLabelings:

            # probability of paths ending with a non-blank
            prNonBlank = 0
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

            # probability of paths ending with a blank
            prBlank = (last.entries[labeling].prTotal) * mat[t, blankIdx]

            # add beam at current time-step if needed
            addBeam(curr, labeling)

            # fill in data
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].prNonBlank += prNonBlank
            curr.entries[labeling].prBlank += prBlank
            curr.entries[labeling].prTotal += prBlank + prNonBlank
            curr.entries[labeling].prText = last.entries[labeling].prText # beam-labeling not changed, therefore also LM score unchanged from
            curr.entries[labeling].lmApplied = True # LM already applied at previous time-step for this beam-labeling

            # extend current beam-labeling
            for c in range(maxC - 1):
                # add new char to current beam-labeling
                newLabeling = labeling + (c,)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    prNonBlank = mat[t, c] * last.entries[labeling].prBlank
                else:
                    prNonBlank = mat[t, c] * last.entries[labeling].prTotal

                # add beam at current time-step if needed
                addBeam(curr, newLabeling)
                
                # fill in data
                curr.entries[newLabeling].labeling = newLabeling
                curr.entries[newLabeling].prNonBlank += prNonBlank
                curr.entries[newLabeling].prTotal += prNonBlank
                
                # apply LM
                applyLM(curr.entries[labeling], curr.entries[newLabeling], classes, lm)

        # set new beam state
        last = curr

    # normalise LM scores according to beam-labeling-length
    last.norm()

    #  # sort by probability
    # bestLabeling = last.sort()[0] # get most probable labeling

    # # map labels to chars
    # res = ''
    # for l in bestLabeling[0]:
    #     res += classes[l]

    return last.sort()[:10]


### Added by MaterialEyes

def get_legal_next_characters(path, sequence_length=8):
    path_length = len(path)
    spots_left = sequence_length - path_length
    if path_length == 0:
        return [0,1,2,3,4,5,6,7,8,9]
    prefix = False
    base_unit = False
    nonzero_digits = 0
    digits = 0
    decimals = 0

    for label in path: 
        if label in [1,2,3,4,5,6,7,8,9]:
            nonzero_digits += 1
            digits += 1
        elif label == 0:
            digits += 1
        elif label == 19:
            decimals += 1
        elif label in [10,11,12,13,14,15,16,17] and not prefix:
            prefix = True
        elif label == 20:
            prefix = True
            base_unit = True
        elif label in [10, 11] and prefix:
            base_unit = True
        elif label in [18, 21, 22]:
            continue
        else:
            print("How did I get here?\nThe path is: ", path)
    
    # unit has been started, no digits or decimals allowed
    if prefix:
        # unit has not been finished, no prefixes allowed
        if not base_unit:
            # only one spot left, must finish unit
            if spots_left == 1:
                return [10, 11]
            else:
                return [10, 11, 18, 21] 
        # unit has been finished, only blanks left
        else:
            return [18, 21]
    # unit has not been started
    # decimal must be followed by a digit
    if label == 19:
        return [0,1,2,3,4,5,6,7,8,9,18,21]
    # if unit hasn't started and only one spot left, must be A
    if spots_left == 1:
        return [20]
    elif spots_left == 2:
        # current label is a space, can go right into unit
        if label in [18,21]:
            return [10,11,12,13,14,15,16,17,18,20,21]
        else:
            return [18, 21]
    # more than 2 spots left
    # if last spot is not a blank, must be followed by more numbers or spaces
    if label not in [18, 21]:
        if decimals == 1:
            return [0,1,2,3,4,5,6,7,8,9,18,21]
        else:
            return [0,1,2,3,4,5,6,7,8,9,19,18,21]
    # last spot is blank, can be followed by anything
    if decimals == 1:
        return [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21]
    else:
        return [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
 
def postprocess_ctc(results):
    classes = "0123456789mMcCuUnN .A"
    idx_to_class = classes + "-"
    for result, confidence in results:
        confidence = float(confidence)
        word = ""
        for step in result:
            word += idx_to_class[step]
        word = word.strip()
        word = "".join(word.split("-"))
        try:
            number, unit = word.split()
            number = float(number)
            if unit.lower() == "n":
                unit = "nm"
            elif unit.lower() == "c":
                unit = "cm"
            elif unit.lower() == "u":
                unit = "um"
            if unit.lower() in ["nm", "mm", "cm", "um", "a"]:
                return number, unit, confidence
        except Exception as e:
            continue
    return -1, "m", 0

def run_ctc(probs, classes):
    current_file = pathlib.Path(__file__).resolve(strict=True)
    language_model_file ="corpus.txt"
    language_model = LanguageModel(current_file.parent / language_model_file, classes)
    top_results = ctcBeamSearch(probs, classes, lm=language_model, beamWidth=15)
    magnitude, unit, confidence = postprocess_ctc(top_results)
    return magnitude, unit, confidence