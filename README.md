# ClasSeg - An Extendable Classification/Segmentation/SSL DL Pipeline
ClasSeg is a pipeline which can handle multiple tasks and data types out of the box. This is a solid foundation for writing any type of deep learning project.
Still in an alpha stage of development.
# Default Supported Data Types
1) Natural images (PNG, JPG, ....) grayscale and RGB
2) Medical images (Dicom, NIfTI) 2d and 3d
3) Easily extended with a new reader/writer class (and perhaps a change or two to the datapoint object)
# Setup
To get started by trianing an mnist classification follow these steps:
1) Clone this repository ```git clone https://github.com/aheschl1/ClasSegPipeline```
2) Navigate to the root directory ```cd ClasSegPipeline```
3) Install the package ```pip install -e .```
4) Preprocess the dataset (we have written an extension to manage this simple use-case) ```classegPreprocess -d <any integer id number <= 3 digits> -dd <text description> -f <number of folds> -ext mnist_class```
5) Train the dataset ```classegTrain -d <dataset id> -f <fold to train> -ext mnist_class -m efficientnetb0_one_channel -n <experiment name>```

**What is created on the file system?**
1) When you ran the first command, ClasSeg created a configuration file at ```~/.classegrc```
2) Dataset roots have been created: ```~/Documents/Datasets/ClasSegRoot/raw```, ```~/Documents/Datasets/ClasSegRoot/preprocessed```, ```~/Documents/Datasets/ClasSegRoot/results``` (You can update where these should be in the ~/.classegrc file)
3) When you preprocessed: "<pwd>/MNIST" was temporarily created, and then deleted by the proprocessor. This is where the data was downloaded to.
4) When you preprocessed: ```RAW_ROOT/Dataset_<desc>_<id>```, ```PREPROCESSED/Dataset_<desc>_<id>```. Take a look at what information is avaialble, and the default config files.
5) When you trained: ```RESULTS_ROOT/Dataset_<desc>_<id>/fold_<fold>/<experiment name>```. This is where you can find logs, **tensorboard logs**, weights, and some backups.

# Notable Out of the Box Features
1) Multiple mode training with no coding needed. Mode is determined by file system structure.
2) Model customization and design with no coding (though it can be desireable to code for more complex systems) thanks to https://github.com/aheschl1/JsonTorchModels
3) Tensorboard logging
4) Extension capacity for extending functionality, while not touching the core codebase.
5) DDP Training (even with custom trainers)
6) K-Fold training and validation

# How to setup a dataset for preprocessing and trianing?

This is the most involved step for a custom dataset. There are three dataset structures.
_Case numbers can be any number of digits >= 5. Most common extensions are supported. Data can be images or volumes._

**Classification Structure**
```
RAW_ROOT
|
---Dataset_<desc>_<id>
.              |
.              ---- <label_0>
.              .        | case_00000.png
.              .        | case_000101.png
.              ---- <label_1>
.              .        | case_xxxxx.png
.              .        | case_xxxxx.png
.              ---- <label_n>
---Dataset_<desc2>_<id2>
|
```

**SSL Structure**
```
RAW_ROOT
|
---Dataset_<desc>_<id>
.        | case_00000.png
.        | case_000101.png
.        | case_xxxxx.png
.        | case_xxxxx.png
---Dataset_<desc2>_<id2>
|
```

**Segmentation Structure**
```
RAW_ROOT
|
---Dataset_<desc>_<id>
.              |
.              ---- imagesTr
.              .        | case_00000.png
.              .        | case_000101.png
.              ---- labelsTr
.              .        | case_00000.png
.              .        | case_00101.png
---Dataset_<desc2>_<id2>
|
```
# Writing an Extension
Extensions currently allow you to easily create a custom trainer and a preprocessor. More extensionility coming soon!
1) Run **classegCreateExtension**
2) Follow the prompts
3) It is now created at ```<repo_root>/extensions/<extension_name>```
   
To use your new extension, pass the name you chose to the -extension/-ext argument for trianing and preprocessing.

**Trainer**

The trainer is at ```<extension_root>/training/trainer.py```. Check out some precooked extensions for how to develop a new one.

**Preprocessor**

The preprocessor is at ```<extension_root>/preprocessing/preprocessor.py```. Check out some precooked extensions for how to develop a new one.

**Change Trainer/Preprocessor Class Names**

There are default names for the template classes, of course. To rename them, you need to modify **<extension_root>/__init__.py**. Modify TRAINER_CLASS_NAME and PREPROCESSOR_CLASS_NAME as needed.
