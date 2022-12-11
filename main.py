import numpy as np
import pandas as pd
import time
import HKPCoherence

import utils


def distortion_p(dataset, public_items, private_items, h, k, p_list, sigma):
    print(f"\n**************** DISTORTION VS P ****************\n")

    for p in p_list:
        print(f"\n-------------- DISTORTION FOR P: {p} --------------\n")

        hkp = HKPCoherence.HKPCoherence(dataset, public_items, private_items, h, k, p=p, sigma=sigma)

        print(f"Private items: {private_items}\nPrivate len: {len(private_items)}\n\n"
              f"Public items: {public_items}\nPublic len: {len(public_items)}")
        # start the anonymization process
        hkp.execute_algorithm()


def distortion_sigma(dataset, all_items, sigma_list, h, k, p):
    print(f"\n**************** DISTORTION VS SIGMA ****************\n")

    for sigma in sigma_list:
        print(f"\n-------------- DISTORTION FOR SIGMA: {sigma} --------------\n")
        # slice SIGMA percent of the items as public items
        public_items = all_items[:int(len(all_items) * SIGMA)]

        # determine the private items
        private_items = [i for i in all_items if i not in public_items]

        print(f"Private items: {private_items}\nPrivate len: {len(private_items)}\n\n"
              f"Public items: {public_items}\nPublic len: {len(public_items)}")

        # create the hkp object
        hkp = HKPCoherence.HKPCoherence(dataset, public_items, private_items, h, k, p, sigma=sigma)

        # start the anonymization process
        hkp.execute_algorithm()


def distortion_k(dataset, public_items, private_items, h, k_list, p, sigma):
    print(f"\n**************** DISTORTION VS K ****************\n")

    for k in k_list:
        print(f"\n-------------- DISTORTION FOR K: {k} --------------\n")
        print(f"Private items: {private_items}\nPrivate len: {len(private_items)}\n\n"
              f"Public items: {public_items}\nPublic len: {len(public_items)}")

        hkp = HKPCoherence.HKPCoherence(dataset, public_items, private_items, h, k=k, p=p, sigma=sigma)

        # start the anonymization process
        hkp.execute_algorithm()


# TODO: Convert main into a function so it is easier to make multiple runs back to back in order to get the plots
if __name__ == "__main__":
    DATA_PATH = "./Dataset/T40I10D100K_100.txt"

    p_list = [2, 3, 4, 5, 6, 7]
    k_list = [5, 10, 20, 30, 40, 50]
    sigma_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30]

    # STANDARD PARAMETERS
    H = 0.4
    K = 10
    P = 3
    SIGMA = 0.15  # percentage of public items selected from the dataset

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

    # distortion_p(dataset, public_items, private_items, h=H, k=K, p_list=p_list, sigma=SIGMA)
    # distortion_sigma(dataset, all_items=unique_items, sigma_list=sigma_list, h=H, k=K, p=P)
    # distortion_k(dataset, public_items, private_items, h=H, k_list=k_list, p=P, sigma=SIGMA)

    # create the hkp object
    hkp = HKPCoherence.HKPCoherence(dataset, public_items, private_items, h=H, k=K, p=P, sigma=SIGMA)

    # start the anonymization process
    hkp.execute_algorithm()

    hkp.anonymization_verifier()
