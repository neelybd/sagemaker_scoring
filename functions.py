from collections import OrderedDict
import numpy as np
from dateutil.parser import parse


# print("Function: Functions")
# print("Release: 1.1.1")
# print("Date: 2020-06-26")
# print("Author: Brian Neely")
# print()
# print()
# print("General Functions")
# print()
# print()


def is_date(string, fuzzy=False):
    """
    Created by Alex Riley
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def dedupe_list(duplicate_list):
    # *****Dedup list*****
    return list(OrderedDict.fromkeys(duplicate_list))


def split_data(data, num_splits):
    # *****Split data for parallel processing*****
    # Calculate the split locations
    split_locations = np.linspace(0, len(data), num_splits)
    # Rounds up the  split_locations
    split_locations = np.ceil(split_locations)
    # Convert split_locations to int for splitting data
    split_locations = split_locations.astype(int)
    # Split data for parallel processing
    data_split = np.split(data, split_locations)

    return data_split


def list_diff(list_1, list_2):
    # Return different items between lists
    return [i for i in list_1 + list_2 if i not in list_1 or i not in list_2]


def list_common(list_1, list_2):
    # Return common items between lists
    return list(set(list_1).intersection(list_2))
