import itertools


def find_subarrays_of_size_n(l: list, n: int):
    """

    :param l: The list of elements
    :param n: Number of elements in a sub-array
    :return: A map object containing the n-length sub-arrays
    """
    return itertools.combinations(l, n)


def find_unique_items(input_file):
    with open(input_file, "r") as file:
        lines = file.read()
        unique_items = set(lines.split())

        return unique_items
