import numpy as np
import time
import itertools
import pickle
import HKPCoherence

from anytree import NodeMixin, RenderTree, search

import utils

if __name__ == "__main__":
    DATA_PATH = "./Dataset/T40I10D100K_100.txt"

    # find unique items to randomly choose the private items among them
    unique_items = utils.find_unique_items(DATA_PATH)
    unique_items = [int(i) for i in unique_items]
    print(f"Number of Unique items: {len(unique_items)}")

    # randomly choose private items
    # dataset =
    np.random.seed(42)
    private_items = np.random.choice(unique_items, replace=False, size=10)
    public_items = [i for i in unique_items if i not in private_items]
    public_items = public_items[:50]
    print(f"private: {len(private_items)}\npublic: {len(public_items)}")

    dataset = []
    with open(DATA_PATH, "r") as file:
        for line in file:
            dataset.append([int(i) for i in set(line.rstrip().split())])

    # create an obj
    hkp = HKPCoherence.HKPCoherence(dataset, public_items, private_items, h=0.8, k=2, p=2)

    # start the anonymization process
    hkp.execute_algorithm()

    # pickle the hkp object for later use
    with open("hkp_complete_object.pkl", "wb") as f:
        pickle.dump(hkp, f)