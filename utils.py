import itertools
import pandas as pd
from anytree import search


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


# check if MM values in score table equal total mole_num count
# if there is inequality raises assertion error
def check_MM_equal_mole_num(tree_root, score_table):
    for index, item in enumerate(score_table.keys()):
        count = 0
        test_list = search.findall(tree_root, filter_=lambda node: node.label == item)
        for i in test_list:
            count += i.mole_num
        assert count == score_table[item][
            'MM'], f"{index} NOT EQUAL -- Item: {item}, Score_table MM: {score_table[item]['MM']}, " \
                   f"mole num count: {count}"
        if count != score_table[item]['MM']:
            print(f"{index} NOT EQUAL -- Item: {item}, Score_table MM: {score_table[item]['MM']}, "
                  f"mole num count: {count}")
        else:
            print(
                f"{index} -- Item: {item}, Score_table MM: {score_table[item]['MM']}, mole num count: {count}")


def convert_txt_to_csv(input_txt, output_csv):
    dataset = []
    with open(input_txt, "r") as file:
        for line in file:
            dataset.append([int(i) for i in set(line.rstrip().split())])

    data_df = pd.DataFrame(dataset)
    print(data_df.head())
    data_df.to_csv(output_csv, index=False, header=False)


def cut_txt_file(input_file, output_file, n: int):
    # funcntion for cutting first n lines of a text file
    # first n lines written into a specified output file for later use
    with open(input_file, "r") as original_file, open(output_file, "w") as cut_file:
        head = [next(original_file) for x in range(n)]
        for line in head:
            cut_file.write(line)
