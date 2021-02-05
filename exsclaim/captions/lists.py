import collections

def is_disjoint(l1,l2) -> bool:
    """ Determines if two lists share any common elements

    :param l1: List 1
    :param l2: List 2

    :return: A bool determining if lists overlap at all
    """
    return len(set(l1).intersection(set(l2))) == 0

def intersection(lst1, lst2):
    """ Returns all elements in both list1 and list2 """
    list_intersection = [value for value in lst1 if value in lst2] 
    return list_intersection
    
def flatten(items: list) -> list:
    """ Yield items from any nested iterable; 

    :param items: A nested iterable (list)

    :return: A flattened list
    """
    for x in items:
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x