import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--experiment', type=str,
                    help="Specify the path of the images dataset from the current location.")

def performance_plot(args):
    """
    To plot the loss and f1-scores from the csv files saved by save_track()
    """
    df_train = pd.read_csv("../experiments/" + args.experiment + "/" + args.experiment + "_train_tracking.csv")
    df_val = pd.read_csv("../experiments/" + args.experiment + "/" + args.experiment + "_val_tracking.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(df_train['loss'], label='train loss')
    ax2.plot(df_train['f1_patch'], label='train patch F1-score')
    ax1.plot(df_val['loss'], label='val loss')
    ax2.plot(df_val['f1_patch'], label='val patch F1-score')
    ax1.legend()
    ax2.legend()
    ax1.set_title("Training and validation loss")
    ax2.set_title("Training and validation F1-score")
    ax1.grid()
    ax2.grid()
    ax1.set_xlabel("Epochs")
    ax2.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("F1-score")
    fig.suptitle("Training and validation loss and F1-score for experiment " + args.experiment)
    plt.savefig("../experiments/" + args.experiment + "/" + args.experiment + "_performances.png")
    plt.show()

if __name__ == '__main__':
     # Getting the arguments
    args = parser.parse_args()

    # Validating the arguments
    if args.experiment is None:
        raise Exception("Please specify the name of the experiment you want to plot.")

    performance_plot(args)