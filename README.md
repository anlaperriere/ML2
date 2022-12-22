# Road Segmentation project


This repository contains the work achieved by Salomé BAUP, Tanguy DESJARDIN and Antoine LAPERRIERE during the fall 2022 semester for the second project of the CS-433 EPFL Machine Learning course.
\
The goal of the project is to identify roads on satellite images extracted from google maps using a machine learning model. The training dataset consists of images and their corresponding groundtruth where the road corresponds to white pixels and the rest is black.
<p align="center">
<img src="data/test_set_images/test_42/test_42.png" alt="classdiagram"  width="200" title="hover text">
<img src="data/Images_readme/UNET.png"  alt="classdiagram" width="200" >
<img src="data/Images_readme/ResNet.png"  alt="classdiagram" width="200" >
<figcaption align = "center"><b>Fig.1 - Testing image and prediction (UNet on the left, UNet with ResNet encoder on the right)</b></figcaption>
</p>

The train and test datasets are stored in the `data/` folder.

## Environnement setup
In order to reproduce our predictions, you can install the required libraries using the following command:
\
(note that the trainings have been performed on macOS with a M1 Max GPU using python 3.9.15 and on Windows with a Nvidia GPU using python 3.9.13)

Run the following command to create a conda virtual environment:
```bash
conda create --name road_segmentation python=3.9.15
conda activate road_segmentation   
```

Head to the folder you want to contain this GitHub repository:
```bash
cd my/folder/path
```

Clone this GitHub repository:
```bash
git clone https://github.com/CS-433/ml-project-2-magical_mandrills/
```

Move into the cloned repo:
```bash
cd ml-project-2-magical_mandrills/
```

If you are using MacOS with an ARM architecture run:

```bash
pip install -r requirements_mac_m1.txt
```

If you are using Windows with a Nvidia GPU run:

```bash
pip install -r requirements_windows_cuda.txt
```

Move into scripts folder:
```bash
cd scripts
```
## Scripts
The `scripts` folder contains the following files:
* main.py: contains the code to train and test the model
* datasets.py: contains the code to load the dataset and augment it
* model.py: contains the code for UNet model
* helper.py: contains the code to compute the metrics and to save the predictions
* performance.py: contains the code to compute the performances of the model
* PostProcessMO.ipynb: contains morphological operations post processing work

The main.py script contains the code to train and test the model. It will use all the other scripts to get predictions (except the .ipynb file). The arguments are the following:

| Flag                  | Type             | Default                    |Description                                                                    | 
| --------------------- |------------------|----------------------------|-------------------------------------------------------------------------------|
| data_path             | str              | "../data"                  | Dataset path.                                                                 |
| train                 | ast.literal_eval | False                      | Train the model if true                                                       |
| test                  | ast.literal_eval | True                       | Make prediction on test set itrue                                             |
| validation_ratio      | float            | 0.2                        | Set validatioratio                                                            |
| device                | str              | "cpu"                      | Choose the device on which to train the model (cpu, cuda or mps)              |
| lr                    | float            | 0.001                      | Set the learningrate                                                          |
| experiment_name       | str              | "Best_model"               | Name of the experiment (Important: do not put figures/numbers in the name)    |
| weights_path          | str              | "../experiments/R_K/R_K.pt"| Name of the experiment (Important: do not put figures/numbers in the name)    |
| loss                  | str              | "dice"                     | Choose the type of loss (dice, BCE or diceBCE)                                |
| save_weights          | ast.literal_eval | True                       | Save weights after every epoch if True                                        |
| epochs                | int              | 100                        | Set the number ofepochs                                                       |
| model                 | str              | "ResNet50"                 | Set the model on which to train (UNet or ResNet50)                            |
| flip                  | ast.literal_eval | True                       | Flip images if true (horizontally and vertically)                             |
| rotation              | ast.literal_eval | True                       | Rotate images if true                                                         |
| grayscale             | ast.literal_eval | True                       | Apply grayscale on some images if true                                        |
| erase                 | int              | 5                          | Number of random erase per images                                             |
| batch_size            | int              | 32                         | Select the batch size                                                         |
| opti                  | str              | "Adam"                     | Select the optimizer (Adam or Adamax)                                         |
\
## Test our best model
\
To obtain the predictions achieved with our best model, you can use our pretrained models. The best pretrained model is stored on this [Google Drive](https://drive.google.com/drive/folders/1DNUKSZgf0mBA7StU-iCSydfxbGH_1LL4?usp=sharing). The model is located in `R_K` folder. You can download the `R_K.pt` file and place it in your local `experiments/R_K` folder. Then run the following command:

From mac0S with MPS:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -W ignore main.py --device "mps"
```

From windows with CUDA:
```bash
python main.py --device "cuda"
```

## Retrain our best model
\
To retrain the best model we obtained you can run the following command on mac0S:
(note that the training took approximately 2h30 on a M1 max 32GC 32 GB RAM MacBook Pro)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -W ignore main.py --train True --device "mps"
```

or on Windows:
```bash
python main.py --train True --device "cuda"
```

Once the computation is done, the prediction and this best models parameters are stored in the `Best_model` folder.

## Train and test your own model
\
If you want to train the model from scratch, here is an example command for macOS:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -W ignore main.py --experiment_name "my_own_model" --data_path "../data" --save_weights True --device "mps" --model "ResNet50" --train True --test True --validation_ratio 0.2 --epochs 50 --lr 0.0001 --loss "cross entropy" --opti "Adam" --flip True --rotation True --grayscale True --erase 5 --save_weights True --batch_size 16
```

Or for windows:
```bash
python main.py --experiment_name "my_own_model" --data_path "../data" --save_weights True --device "cuda" --model "ResNet50" --train True --test True --validation_ratio 0.2 --epochs 50 --lr 0.0001 --loss "cross entropy" --opti "Adam" --flip True --rotation True --grayscale True --erase 5 --save_weights True --batch_size 16
```

## Evaluate performances of a pretrained model
\
If you want to evaluate the performances (loss and F1-score) of a given pretrained experiment, load the complete folder (R_x or U_x) from the Google Drive into `experiments/` and execute the performance script with:
```bash
python performance.py --experiment "R_x"
```

# Contributors
* Salomé BAUP
* Tanguy DESJARDIN
* Antoine LAPERRIERE

# Citation
```bash
@misc{MMLS,
  Author = {Salomé BAUP, Tanguy DESJARDIN \& Antoine LAPERRIERE},
  Title = {Road segmentation on satellite images},
  Year = {2022},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/CS-433/ml-project-2-magical_mandrills}}
```
