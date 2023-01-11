import itertools
import pandas as pd
import numpy as np
from anytree import search
import HKPCoherence


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


def distortion_p(data_path, h, k, p_list, sigma):
    print(f"\n**************** DISTORTION VS P ****************\n")

    for p in p_list:
        dataset, public_items, private_items = prepare_data(data_path, sigma=sigma)
        print(f"\n-------------- DISTORTION FOR P: {p} --------------\n")

        hkp = HKPCoherence.HKPCoherence(dataset, public_items, private_items, h, k, p=p, sigma=sigma)

        # start the anonymization process
        hkp.execute_algorithm(check_verification=True)


def distortion_sigma(data_path, sigma_list, h, k, p):
    print(f"\n**************** DISTORTION VS SIGMA ****************\n")

    for sigma in sigma_list:
        print(f"\n-------------- DISTORTION FOR SIGMA: {sigma} --------------\n")

        dataset, public_items, private_items = prepare_data(data_path, sigma=sigma)

        # create the hkp object
        hkp = HKPCoherence.HKPCoherence(dataset, public_items, private_items, h, k, p, sigma=sigma)

        # start the anonymization process
        hkp.execute_algorithm(check_verification=True)


def distortion_k(data_path, h, k_list, p, sigma):
    print(f"\n**************** DISTORTION VS K ****************\n")

    for k in k_list:
        dataset, public_items, private_items = prepare_data(data_path, sigma=sigma)
        print(f"\n-------------- DISTORTION FOR K: {k} --------------\n")

        hkp = HKPCoherence.HKPCoherence(dataset, public_items, private_items, h, k=k, p=p, sigma=sigma)

        # start the anonymization process
        hkp.execute_algorithm(check_verification=True)


def prepare_data(data_path, sigma):
    # set a random seed
    np.random.seed(10)

    # make a list of all items that can be found the dataset
    unique_items = find_unique_items(data_path)
    unique_items = [int(i) for i in unique_items]
    print(f"Number of Unique items: {len(unique_items)}")

    # shuffle all items
    unique_items = np.random.permutation(unique_items)

    # slice SIGMA percent of the items as public items
    public_items = unique_items[:int(len(unique_items) * sigma)]

    # determine the private items
    private_items = [i for i in unique_items if i not in public_items]

    print(f"Sigma: {sigma*100}%\nPrivate items count: {len(private_items)}\n"
          f"Public items count: {len(public_items)}\n\nPublic items list: {public_items}")

    # read all the data from the text file
    dataset = []
    with open(data_path, "r") as file:
        for line in file:
            dataset.append([int(i) for i in set(line.rstrip().split())])

    return dataset, public_items, private_items

def node_link_length(head_link):
    """
    Finds the length of a node_link

    :param head_link: Head node of the node_link
    :return:
    """
    temp = head_link  # Initialise temp
    count = 0  # Initialise count

    if not head_link:
        return count

    else:
        # Loop while end of linked list is not reached
        while temp:
            count += 1
            temp = temp.node_link
        return count