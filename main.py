import numpy as np
import time
import itertools
import pickle
import HKPCoherence

from anytree import NodeMixin, RenderTree, search

import utils

if __name__ == "__main__":
    DATA_PATH = "./Dataset/T40I10D100K_1000.txt"

    # PARAMETERS
    H = 0.4
    K = 30
    P = 2
    SIGMA = 0.04  # percentage of public items selected from the dataset

    # set a random seed
    np.random.seed(42)

    # make a list of all items that can be found the dataset
    unique_items = utils.find_unique_items(DATA_PATH)
    unique_items = [int(i) for i in unique_items]
    print(f"Number of Unique items: {len(unique_items)}")

    # shuffle all items
    unique_items = np.random.permutation(unique_items)

    # slice SIGMA percent of the items as public items
    public_items = unique_items[:int(len(unique_items) * SIGMA)]

    # determine the private items
    private_items = [i for i in unique_items if i not in public_items]

    print(f"Private items: {private_items}\nPrivate len: {len(private_items)}\n\n"
          f"Public items: {public_items}\nPublic len: {len(public_items)}")

    # read all the data from the text file
    dataset = []
    with open(DATA_PATH, "r") as file:
        for line in file:
            dataset.append([int(i) for i in set(line.rstrip().split())])

    # create the hkp object
    hkp = HKPCoherence.HKPCoherence(dataset, public_items, private_items, h=H, k=K, p=P)

    # start the anonymization process
    hkp.execute_algorithm()
    hkp.anonymization_verifier()

    # pickle the hkp object for later use
    with open("hkp_complete_object.pkl", "wb") as f:
        pickle.dump(hkp, f)
