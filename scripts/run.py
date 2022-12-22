import datasets
import ast
import segmentation_models_pytorch as smp
from model import UNet
from torch.utils.data import DataLoader
from helpers import *

EXPREMINENT_NAME = "Best_model"
DATA_PATH = "../data"
WEIGHTS_PATH = "../experiments/R_K/R_K.pt"
DEVICE = "cpu"
TRAIN = False
BATCH_SIZE = 32
LR = 0.001

def run(EXPREMINENT_NAME, DATA_PATH, WEIGHTS_PATH, DEVICE, BATCH_SIZE):

    # Ensure reproducibility
    seed = 5
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Experiment folder creation
    experiment_path = os.path.join("../experiments", EXPREMINENT_NAME)
    create_folder(experiment_path)

    # Test dataset
    test_dataset = datasets.DatasetTest(path=DATA_PATH)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model initialization
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
    model = model.to(DEVICE)

    # Optimizer initialization
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)

    # Loading previous state for model weights and optimizer
    load_model(model, optimizer, DEVICE, WEIGHTS_PATH)
    print("Loaded weights")

    # Plateau scheduler initialization
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, min_lr=1e-7
    )

    # Loss function initialization
    criterion = dice_loss

    # Testing the model
    # Folder to save output images and predictions
    results_path = os.path.join(experiment_path, "results")
    create_folder(results_path)

    # If training was performed, loads the best model obtained after this training
    model.eval()
    with torch.no_grad():
        for i, img in enumerate(test_loader):

            img = img.to(DEVICE).float()
            output = model(img)

            # Outputs masks saved as images
            save_image(output, i + 1, results_path)
            save_image_overlap(output, img, i + 1, results_path)

    # Converting the saved masks to a submission file
    submission_filename = os.path.join(results_path, EXPREMINENT_NAME + ".csv")
    image_filenames = []
    for i in range(1, 51):
        image_filename = results_path + "/satImage_" + "%.3d" % i + ".png"
        print(image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
    print("Testing completed.")


if __name__ == "__main__":

    run(EXPREMINENT_NAME, DATA_PATH, WEIGHTS_PATH, DEVICE, BATCH_SIZE)
