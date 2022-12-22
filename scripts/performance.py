import argparse
import pandas as pd
import matplotlib.pyplot as plt
from helpers import performance_plot

"""
This script is used to plot the performance of a given experiment (loss and f1-score) via the command line.
"""
parser = argparse.ArgumentParser()

parser.add_argument(
    "--experiment",
    type=str,
    help="Specify the path of the images dataset from the current location.",
)

if __name__ == "__main__":
    # Getting the arguments
    args = parser.parse_args()

    # Validating the arguments
    if args.experiment is None:
        raise Exception("Please specify the name of the experiment you want to plot.")

    performance_plot(args.experiment)
