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
    h = args.h_value
    k = args.k_value
    p = args.p_value
    sigma = args.sigma
    data_path = args.data_path
    verification_flag = args.no_verification

    dataset, public_items, private_items = utils.prepare_data(data_path, sigma)

    # create the hkp object
    hkp = HKPCoherence.HKPCoherence(dataset, public_items, private_items, h, k, p, sigma)

    # start the anonymization process
    hkp.execute_algorithm(check_verification=verification_flag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs (h,k,p)-coherence algorithm',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-hv', '--h-value',
                        type=percentage,
                        default=0.4,
                        help='(0,1) percentage value. (default: %(default)s)')

    parser.add_argument('-k', '--k-value',
                        type=positive_int,
                        default=10,
                        help='enter the value of k, positive integer. (default: %(default)s)',
                        )

    parser.add_argument('-p', '--p-value',
                        type=positive_int,
                        default=4,
                        help='enter the value of p, positive integer. (default: %(default)s)'
                        )

    parser.add_argument('-s', '--sigma',
                        type=percentage,
                        default=0.15,
                        help='(0,1) percentage value. (default: %(default)s)'
                        )

    parser.add_argument('-d', '--data-path',
                        type=str,
                        default='Dataset/T40_1000.txt',
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
