import utils

if __name__ == "__main__":
    # full set 'Dataset/T40I10D100K.dat'
    DATA_PATH = 'Dataset/T40I10D100K.dat'

    p_list = [2, 3, 4, 5, 6, 7]
    k_list = [5, 10, 20, 30, 40, 50]
    # sigma_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30]

    # STANDARD PARAMETERS
    H = 0.4
    K = 30
    P = 4
    SIGMA = 0.15  # percentage of public items selected from the dataset

    utils.distortion_p(DATA_PATH, h=H, k=K, p_list=p_list, sigma=SIGMA)
    # utils.distortion_sigma(DATA_PATH, sigma_list=sigma_list, h=H, k=K, p=P)
    # utils.distortion_k(DATA_PATH, h=H, k_list=k_list, p=P, sigma=SIGMA)

    # for i in range(4):
    #     utils.distortion_k(DATA_PATH, h=H, k_list=k_list, p=P, sigma=SIGMA)
    #
    # for i in range(4):
    #     utils.distortion_p(DATA_PATH, h=H, k=K, p_list=p_list, sigma=SIGMA)

