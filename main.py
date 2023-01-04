
import HKPCoherence

import utils

if __name__ == "__main__":
    DATA_PATH = "./Dataset/T40I10D100K_5000.txt"

    # p_list = [2, 3, 4, 5, 6, 7]
    # k_list = [5, 10, 20, 30, 40, 50]
    # sigma_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30]

    # STANDARD PARAMETERS
    H = 0.4
    K = 5
    P = 2
    SIGMA = 0.15  # percentage of public items selected from the dataset

    # distortion_p(DATA_PATH, h=H, k=K, p_list=p_list, sigma=SIGMA)
    # distortion_sigma(DATA_PATH, sigma_list=sigma_list, h=H, k=K, p=P)
    # distortion_k(DATA_PATH, h=H, k_list=k_list, p=P, sigma=SIGMA)

    dataset, public_items, private_items = utils.prepare_data(DATA_PATH, sigma=SIGMA)

    # create the hkp object
    hkp = HKPCoherence.HKPCoherence(dataset, public_items, private_items, h=H, k=K, p=P, sigma=SIGMA)

    # start the anonymization process
    hkp.execute_algorithm()
