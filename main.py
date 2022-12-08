import argparse
import datasets
from models import UNet
from torch.utils.data import DataLoader
from helpers import *
import ast
import segmentation_models_pytorch as smp

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data',
                    help="Dataset path")
parser.add_argument('--model', type=str, default="UNet",
                    help="Selects the model. Acceptable values: 'UNet' or 'ResNet50'")
parser.add_argument('--validation_ratio', type=float, default=None,
                    help="The ratio of validation dataset size to the whole dataset. \
                    If not set then there will be no validation and the whole dataset is used for training")
parser.add_argument('--rotate', type=ast.literal_eval, default=True,
                    help="Random rotations while training")
parser.add_argument('--flip', type=ast.literal_eval, default=True,
                    help="Random flips while training")
parser.add_argument('--grayscale', type=ast.literal_eval, default=False,
                    help="Random grayscale while training")
parser.add_argument('--random_crops', type=int, default=0,
                    help="Number of random crops for data augmentation")
parser.add_argument('--resize', type=int, default=None,
                    help="The new size for train and validation images.")
parser.add_argument('--batch_size', type=int, default=8,
                    help="The batch size for the training")
parser.add_argument('--device', type=str, default="cpu",
                    help="If mps or cuda, gpu is used for training. Otherwise, training is performed on cpu.)
parser.add_argument('--lr', type=float, default=0.001,
                    help="The learning rate value")
parser.add_argument('--weight_path', type=str, default=None,
                    help="The path to saved weights. if not specified there will be no weight loaded")
parser.add_argument('--experiment_name', type=str, default="NotSpec",
                    help="The name of the experiment")
parser.add_argument('--train', type=ast.literal_eval, default=True,
                    help="If true then training is done")
parser.add_argument('--test', type=ast.literal_eval, default=True,
                    help="If true then test is done")
parser.add_argument('--epochs', type=int, default=100,
                    help="Number of epoch")
parser.add_argument('--save_weights', type=bool, default=False,
                    help="If true the weights are saved, only for the epochs where the model achieves a better validation losses")
parser.add_argument('--loss', type=str, default="dice",
                    help="Selects the loss type. \
                    The accepted values are 'dice', 'cross entropy' and 'dice + cross entropy'")


def main(args):
    
    # Ensure reproducibility
    seed = 5
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Dataset initialization
    ratio = args.validation_ratio if args.validation_ratio else 0
    train_dataset = datasets.DatasetTrainVal(
        path=args.path, set_type='train',
        ratio=ratio, rotate=args.rotate,
        flip=args.flip,
        grayscale=args.grayscale,
        random_crops=args.random_crops,
        resize=args.resize,
    )
    test_dataset = datasets.DatasetTest(path=args.path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    if args.validation_ratio:
        val_dataset = datasets.DatasetTrainVal(
            path=args.path,
            set_type='val',
            ratio=ratio,
            rotate=args.rotate,
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
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    elif args.device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")

    # Optimizer initialization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    # Loading state dict for weights and optimizer state
    if args.weight_path:
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
        if args.device == "cuda" and torch.cuda.is_available():
            criterion = criterion.cuda()
        elif args.device == "mps" and torch.backends.mps.is_available():
            criterion = criterion.to("mps")
    elif args.loss == 'dice + cross entropy':
        ce = torch.nn.BCELoss(reduction='mean')
        if args.device == "cuda" and torch.cuda.is_available():
            ce = ce.cuda()
        elif args.device == "mps" and torch.backends.mps.is_available():
            ce = ce.to("mps")
        criterion = lambda output_, mask_: ce(output_, mask_) + dice_loss(output_, mask_)
    else:
        raise Exception("The given loss function does not exist.")

    # Training
    if args.train:
        best_loss = 1.0
        for epoch in range(args.epochs):
            model.train()
            train_loss = []
            train_f1 = []
            train_f1_patches = []
            for img, mask in train_loader:
                
                if args.device == "cuda" and torch.cuda.is_available():
                    img = img.cuda().float()
                    mask = mask.cuda()
                elif args.device == "mps" and torch.backends.mps.is_available():
                    img = img.to("mps").float()
                    mask = mask.to("mps")
                else:
                    img = img.float()

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
            if args.validation_ratio:
                model.eval()
                best_loss = 1.0
                val_loss = []
                val_f1 = []
                val_f1_patches = []
                with torch.no_grad():
                    for img, mask in val_loader:
                    
                        if args.device == "cuda" and torch.cuda.is_available():
                            img = img.cuda().float()
                            mask = mask.cuda()
                        elif args.device == "mps" and torch.backends.mps.is_available():
                            img = img.to("mps").float()
                            mask = mask.to("mps")
                        else:
                            img = img.float()

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
            else:
                print("Epoch : {} | No validation".format(epoch))

            # Saving the weights
            if args.save_weights and val_loss_to_track < best_loss:
                best_loss = val_loss_to_track
                print('Model_saved at epoch {}'.format(epoch))
                save_model(model, optimizer, experiment_path, args)

    # Testing
    if args.test:
        results_path = os.path.join(experiment_path, 'results')
        create_folder(results_path)
        model.eval()
        with torch.no_grad():
            for i, img in enumerate(test_loader):
                    if args.device == "cuda" and torch.cuda.is_available():
                        img = img.cuda().float()
                    elif args.device == "mps" and torch.backends.mps.is_available():
                        img = img.to("mps").float()
                    else:
                        img = img.float()

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

    # Getting and validating the arguments
    args = parser.parse_args()
    # if args.cuda:
    #     if not torch.cuda.is_available():
    #         raise Exception("GPU not available. Set --cuda False to run with CPU.")
    if args.validation_ratio > 0.8 or args.validation_ratio < 0:
        raise Exception("Validation ratio is not acceptable. Please enter a value between 0 and 0.8.")
    if args.model not in ('UNet', 'ResNet50'):
        raise Exception("The given model does not exist.")
    if args.loss not in ('dice', 'cross entropy', 'dice + cross entropy', 'dice_patches'):
        raise Exception("The given loss function does not exist.\
        Acceptable losses: 'dice', 'cross entropy', 'dice + cross entropy', 'dice_patches'")
    
    main(args)
