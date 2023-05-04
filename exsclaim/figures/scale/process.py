import numpy as np


def non_max_suppression_malisiewicz(boxes, overlap_threshold):
    """Eliminates redundant boxes using NMS adapted from Malisiewicz et al.

    Args:
        boxes (np.ndarray): A >=5 by N array of bounding boxes where each
            of the N rows represents a bounding box with format of
            [x1, y1, x2, y2, confidence, ...]
        overlap_threshold (float): If two boxes exist in which their intersection
            divided by their union (IoU) is greater than overlap_threshold,
            only the box with higher confidence score will remain

    Returns:
        boxes (np.ndarray): A >=5 by K array where K <= N of bounding boxes that
            remain after applying NMS adapted from
        https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
            based off of Malisiewicz et al.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0]))
        )

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]
