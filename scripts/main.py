import argparse
import datasets
import segmentation_models_pytorch as smp
from model import UNet
from torch.utils.data import DataLoader
from helpers import *

# Arguments provided by user
parser = argparse.ArgumentParser()
# Paths
parser.add_argument('--experiment_name', type=str, default="Unidentified_Experiment",
                    help="Specify the name of the current experiment."
                         "It will be used to create a folder to save the results.")
parser.add_argument('--data_path', type=str, default='../data',
                    help="Specify the path of the images dataset from the current location.")
# Training
parser.add_argument('--device', type=str, default="cpu",
                    help="If you want to use a GPU, specify whether it is 'cuda' or 'mps'. Otherwise, CPU is used.")

parser.add_argument('--model', type=str, default="UNet",
                    help="Specify the model. Valid entries: 'UNet' or 'ResNet50'")
parser.add_argument('--train', type=bool, default=True,
                    help="Specify if you want to train the model. Valid entries: True or False")
parser.add_argument('--validation_ratio', type=float, default=0,
                    help="Specify the ratio of data used for validation compared to the whole dataset."
                         "If 0 then all the images are used for training. Valid entries: a number between 0 and 0.5")
parser.add_argument('--batch_size', type=int, default=8,
                    help="Specify the batch size used for training. Valid entries: an integer number")
parser.add_argument('--lr', type=float, default=0.001,
                    help="Specify the learning rate value. Valid entries: a number between 0 and 1")
parser.add_argument('--loss', type=str, default="dice",
                    help="Specify the loss function to use."
                         "Valid entries: 'dice' or 'cross entropy' or 'dice + cross entropy'")
parser.add_argument('--epochs', type=int, default=100,
                    help="Specify the number of epochs the model will be trained on.")
parser.add_argument('--save_weights', type=bool, default=False,
                    help="Specify if you want to save the weights of the trained model. They are progressively saved"
                         "only for the epochs where the model achieves a better validation losses."
                         "Valid entries: True or False")
parser.add_argument('--resize', type=int, default=None,
                    help="If you want to resize images for the training, specify the new size."
                         "Valid entries: an integer number multiple of 32")
# Data augmentation
parser.add_argument('--rotation', type=bool, default=False,
                    help="Specify if you want to augment the data for the training by doing random rotations."
                         "Valid entries: True or False")
parser.add_argument('--flip', type=bool, default=False,
                    help="Specify if you want to augment the data for the training by doing random horizontal"
                         "and vertical flips. Valid entries: True or False")
parser.add_argument('--grayscale', type=bool, default=False,
                    help="Specify if you want to augment the data for the training by randomly gray-scaling images."
                         "Valid entries: True or False")
parser.add_argument('--crops', type=int, default=0,
                    help="Specify how many random crops will be made to augment the data for the training."
                         "Valid entries: an integer number. If you don't want any, enter 0.")
# Testing
parser.add_argument('--test', type=bool, default=True,
                    help="Specify if you want to test the model. Valid entries: True or False")
parser.add_argument('--weights_path', type=str, default=None,
                    help="If you want to test a beforehand trained model, specify the path to the saved weights from"
                         "the current location.")


def main(args):
    
    # Ensure reproducibility
    seed = 5
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Processing unit
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Dataset initialization
    ratio = args.validation_ratio

    train_dataset = datasets.DatasetTrainVal(
        path=args.data_path,
        set_type='train',
        ratio=ratio,
        rotate=args.rotation,
        flip=args.flip,
        grayscale=args.grayscale,
        random_crops=args.crops,
        resize=args.resize,
    )
    test_dataset = datasets.DatasetTest(path=args.data_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    if ratio > 0:
        val_dataset = datasets.DatasetTrainVal(
            path=args.data_path,
            set_type='val',
            ratio=ratio,
            rotate=args.rotation,
            flip=args.flip,
            grayscale=args.grayscale,
            resize=args.resize)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    # Model initialization
    if args.model == 'UNet':
        model = UNet(input_channels=3, output_channels=1)
    elif args.model == "ResNet50":
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )
    else:
        raise Exception("The given model does not exist.")
    
    # Training on GPU if available
    model = model.to(device)

    # Optimizer initialization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    # Loading state dict for weights and optimizer state
    if args.weights_path:
        load_model(model, optimizer, args)

    # Scheduler initialization for reduction of learning rate during the training
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-7)

    # Creating the experiment path
    experiment_path = os.path.join('../experiments', args.experiment_name)
    create_folder(experiment_path)

    # Loss function initialization
    if args.loss == 'dice':
        criterion = dice_loss
    elif args.loss == 'cross entropy':
        criterion = torch.nn.BCELoss(reduction='mean')
        criterion = criterion.to(device)
    elif args.loss == 'dice + cross entropy':
        ce = torch.nn.BCELoss(reduction='mean')
        ce = ce.to(device)
        criterion = lambda output_, mask_: ce(output_, mask_) + dice_loss(output_, mask_)
    else:
        raise Exception("The given loss function does not exist.")

    # Training
    if args.train:
        best_loss = 1.0
        best_epoch = 0
        best_f1_val = 0
        best_f1_train = 0
        for epoch in range(args.epochs):
            model.train()
            train_loss = []
            train_f1 = []
            train_f1_patches = []
            for img, mask in train_loader:

                img = img.to(device).float()
                mask = mask.to(device)

                optimizer.zero_grad()

                # Backward propagation
                output = model(img)
                loss = criterion(output, mask)
                loss.backward()
                optimizer.step()

                f1_score, f1_patches = get_score(output, mask), get_score_patches(output, mask)
                train_loss.append(loss.item())
                train_f1.append(f1_score)
                train_f1_patches.append(f1_patches)
            # Saving the loss and the scores of training for this epoch
            save_track(
                experiment_path,
                args,
                train_loss=sum(train_loss) / len(train_loss),
                train_f1=sum(train_f1) / len(train_f1),
                train_f1_patch=sum(train_f1_patches) / len(train_f1_patches),
            )

            # Validation
            if ratio > 0:
                model.eval()
                val_loss = []
                val_f1 = []
                val_f1_patches = []
                with torch.no_grad():
                    for img, mask in val_loader:
                        img = img.to(device).float()
                        mask = mask.to(device)
                        output = model(img)
                        loss = criterion(output, mask)
                        f1_score, f1_patches = get_score(output, mask), get_score_patches(output, mask)
                        val_loss.append(loss.item())
                        val_f1.append(f1_score)
                        val_f1_patches.append(f1_patches)

                # Logging
                val_loss_to_track = sum(val_loss) / len(val_loss)
                val_f1_to_track = sum(val_f1) / len(val_f1)
                val_f1_patches_to_track = sum(val_f1_patches) / len(val_f1_patches)
                print('Epoch : {} | Loss = {:.4f}, F1 Score = {:.4f}, F1 Patches Score: {:.4f}'.format(
                    epoch, val_loss_to_track, val_f1_to_track, val_f1_patches_to_track))

                # Saving the loss and the scores of validation for this epoch
                save_track(
                    experiment_path,
                    args,
                    val_loss=val_loss_to_track,
                    val_f1=val_f1_to_track,
                    val_f1_patch=val_f1_patches_to_track,
                )

                # Reducing learning rate in case val_loss_to_track does not decrease based on the given patience
                scheduler.step(val_loss_to_track)

                # Saving the weights
                if args.save_weights and val_loss_to_track < best_loss:
                    best_loss = val_loss_to_track
                    best_epoch = epoch
                    best_f1_val = val_f1_to_track
                    best_f1_train = train_f1[-1]
                    print('Model_saved at epoch {}'.format(epoch))
                    save_model(model, optimizer, experiment_path, args)

            else:
                print("Epoch : {} | No validation".format(epoch))
        
        print("The epoch with best_loss is {}, the scores are train_f1 = {:.4f} and val_f1= {:.4f}".format(
            best_epoch, best_f1_train, best_f1_val)
        )

    # If weights path was not specified, load the best model obtained after the current training
    if not args.weights_path:
        if args.train:
            if args.save_weights:
                load_model(model, optimizer, args, os.path.join(experiment_path, args.experiment_name + '.pt'))
            else:
                print("Weight path was not specified nor saved after the training."
                      "Therefore testing is performed on the model at the last epoch, which might not be optimal")
        else:
            print("Weight path was not specified and no training was performed."
                  "Therefore testing is performed but doesn't use a trained model.")
    # Testing
    if args.test:
        results_path = os.path.join(experiment_path, 'results')
        create_folder(results_path)
        model.eval()
        with torch.no_grad():
            for i, img in enumerate(test_loader):

                img = img.to(device).float()

                output = model(img)
                # Saving the output masks
                save_image(output, i + 1, results_path)
                save_image_overlap(output, img, i + 1, results_path)

        # Converting the saved masks to a submission file
        submission_filename = os.path.join(results_path, args.experiment_name + '.csv')
        image_filenames = []
        for i in range(1, 51):
            image_filename = results_path + '/satImage_' + '%.3d' % i + '.png'
            print(image_filename)
            image_filenames.append(image_filename)
        masks_to_submission(submission_filename, *image_filenames)


if __name__ == '__main__':

    # Getting the arguments
    args = parser.parse_args()
    # Validating the arguments
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("You asked for GPU but it is not available. CPU is used instead.")
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            print("You asked for GPU but it is not available. CPU is used instead.")
    if args.device not in ("cuda", "mps", "cpu"):
        raise Exception("Select an appropriate processing unit. You can type help if you don't understand.")
    if args.model not in ("UNet", "Resnet50"):
        raise Exception("Select an appropriate model. You can type help if you don't understand.")
    if args.train not in (True, False):
        raise Exception("Select an appropriate train option. You can type help if you don't understand.")
    if args.validation_ratio > 0.5 or args.validation_ratio < 0:
        raise Exception("Select an appropriate validation ratio. You can type help if you don't understand.")
    if args.batch_size < 0:
        raise Exception("Select an appropriate batch size. You can type help if you don't understand.")
    if args.lr > 1 or args.lr < 0:
        raise Exception("Select an appropriate learning rate. You can type help if you don't understand.")
    if args.loss not in ('dice', 'cross entropy', 'dice + cross entropy'):
        raise Exception("Select an appropriate loss function. You can type help if you don't understand.")
    if args.epochs < 0:
        raise Exception("Select an appropriate number of epochs. You can type help if you don't understand.")
    if args.save_weights not in (True, False):
        raise Exception("Select an appropriate weights saving option. You can type help if you don't understand.")
    if args.resize < 0 or args.resize % 32 != 0:
        raise Exception("Select an appropriate size for image resizing. You can type help if you don't understand.")
    if args.rotation not in (True, False):
        raise Exception("Select an appropriate rotation option. You can type help if you don't understand.")
    if args.flip not in (True, False):
        raise Exception("Select an appropriate flip option. You can type help if you don't understand.")
    if args.grayscale not in (True, False):
        raise Exception("Select an appropriate grayscale option. You can type help if you don't understand.")
    if args.crops < 0:
        raise Exception("Select an appropriate number of random crops. You can type help if you don't understand.")
    if args.test not in (True, False):
        raise Exception("Select an appropriate test option. You can type help if you don't understand.")

    main(args)
