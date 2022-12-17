import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str,
                    help="Specify the name of the experiment you want to plot."
                         "It will be used to create a folder to save the results.")
parser.add_argument('--experiment', type=str,
                    help="Specify the path of the images dataset from the current location.")

def track_plot(args):
    """
    To plot the loss and f1-scores from the csv files saved by save_track()
    """
    df = pd.read_csv(args.path + args.experiment + "/" + args.experiment + "_val_tracking.csv")
    fig, ax = plt.subplots()
    ax.plot(df['loss'], label='val loss')
    ax.plot(df['f1_patch'], label='val f1-score patch')
    plt.ylim(0.2, 0.3)
    ax.legend()
    plt.grid()
    plt.title("Validation loss and f1-score of experiment " + args.experiment)
    plt.xlabel("Epoch")
    plt.ylabel("Loss and f1-score")
    #plt.savefig(args.path + args.experiment + "/" + args.experiment + "_train_tracking.png")
    plt.show()

if __name__ == '__main__':
     # Getting the arguments
    args = parser.parse_args()

    # Validating the arguments
    if args.path is None:
        raise Exception("Please specify the path of the experiment you want to plot.")
    if args.experiment is None:
        raise Exception("Please specify the name of the experiment you want to plot.")

    track_plot(args)