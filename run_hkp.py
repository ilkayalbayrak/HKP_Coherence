import argparse
import HKPCoherence
import utils


def positive_int(s: str) -> int:
    try:
        value = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f'expected integer, got {s!r}')

    if value < 0:
        raise argparse.ArgumentTypeError(f'expected positive integer, got {value}')

    return value


def percentage(s: str) -> float:
    try:
        value = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f'expected float in range (0,1), got {s!r}')

    if 0 < value < 1:
        return value
    else:
        raise argparse.ArgumentTypeError(f'expected float in range (0,1), got {value}')


def main(args):
    h = args.h_val
    k = args.k_val
    p = args.p_val
    sigma = args.sigma
    data_path = args.data_path
    verification_flag = args.no_verification

    dataset, public_items, private_items = utils.prepare_data(data_path, sigma)

    # create the hkp object
    hkp = HKPCoherence.HKPCoherence(dataset, public_items, private_items, h, k, p, sigma)

    # start the anonymization process
    hkp.execute_algorithm(check_verification=verification_flag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs (h,k,p)-coherence algorithm')

    parser.add_argument('--h-val',
                        metavar='float value in range 0,1',
                        type=percentage,
                        default=0.4,
                        help='(0,1) percentage value. (default: %(default)s)')

    parser.add_argument('--k-val',
                        metavar='positive integer',
                        type=positive_int,
                        default=10,
                        help='enter the value of k, positive integer. (default: %(default)s)',
                        )

    parser.add_argument('--p-val',
                        metavar='positive integer',
                        type=positive_int,
                        default=4,
                        help='enter the value of p, positive integer. (default: %(default)s)'
                        )

    parser.add_argument('--sigma',
                        metavar='float value in range 0,1',
                        type=percentage,
                        default=0.15,
                        help='(0,1) percentage value. (default: %(default)s)'
                        )

    parser.add_argument('--data-path',
                        metavar='filepath',
                        type=str,
                        default='Dataset/T40I10D100K_1000.txt',
                        help='enter the path of the dataset. (default: %(default)s)'
                        )

    parser.add_argument('--no-verification',
                        action='store_false',
                        default=True,
                        help='choose if you wanna run the anonymization verifier after anonymizing. '
                             '(default: %(default)s)',
                        )

    args = parser.parse_args()

    # run
    main(args)
