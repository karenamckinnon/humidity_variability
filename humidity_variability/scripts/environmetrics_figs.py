import numpy as np
from humidity_variability.scripts.environmetrics_utils import fit_case
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('this_case', type=int, help='Which case to run (1-4)')
    parser.add_argument('boot_start', type=int, help='Index to start bootstrapping')
    parser.add_argument('nboot', type=int, help='Number of bootstrap samples to calculate')
    parser.add_argument('output_dir', type=str, help='Full path to where to save')
    parser.add_argument('resample_type', type=str, help='Either generative, bootstrap, or jitter')

    args = parser.parse_args()
    lambd_values = np.logspace(-1, 1, 10)
    qs = np.arange(0.05, 1, 0.05)
    qs_int = (100*qs).astype(int)

    fit_case(int(args.this_case), qs, lambd_values, args.boot_start, args.nboot,
             args.output_dir, resample_type=args.resample_type)
