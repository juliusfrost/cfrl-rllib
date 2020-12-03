import argparse

from experiments.utils.checkpoints import get_list_checkpoint


def main():
    """

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--behavior-result-dir', default='results/driving-phase-1',
                        help='directory containing subdirectories with format '
                             'Algorithm_EnvironmentName_id_trial_date_time')

    args = parser.parse_args()
    checkpoints = get_list_checkpoint(args.behavior_result_dir)
    for c in checkpoints:
        print(c)


if __name__ == '__main__':
    main()
