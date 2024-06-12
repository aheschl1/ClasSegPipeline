# ClasSeg - An Extendable Classification/Segmentation/SSL DL Pipeline
ClasSeg is a pipeline which can handle multiple tasks and data types out of the box. This is a solid foundation for writing any type of deep learning project.
This is setup so you only need one codebase for all of your DL projects and research. With capacity to handle multiple datasets with vastly different objectives,
this pipeline is all you need to get efficient training with organized logging, results, and data management.

Still in an early stage of development, but feel free to fork and submit pull requests.

Windows is not compatible currently because of path things. This will be fixed eventually, but Linux > Windows.
# Default Supported Data Types
1) Natural images (PNG, JPG, ....) grayscale and RGB
2) Medical images (Dicom, NIfTI) 2d and 3d
3) Easily extended with a new reader/writer class (and perhaps a change or two to the datapoint object)
# Setup
To get started by trianing an mnist classification follow these steps:
1) Clone this repository ```git clone https://github.com/aheschl1/ClasSegPipeline```
2) Navigate to the root directory ```cd ClasSegPipeline```
3) Install the package ```pip install -e .```
4) Create the environment variables RAW_ROOT, PREPROCESSED_ROOT, RESULTS_ROOT. These should point to folders where:
   1) Raw data will be stored - This is where you will setup your dataset structure. Since this will never be read during training, you can put this on a slower drive.
   2) Preprocessed data will be stored - This is where the preprocessed data will be stored. This is read during training, so it should be on a faster drive.
   3) Results will be stored - This is where the results of training will be stored. This is written to during training.
5) Preprocess the dataset (we have written an extension to manage this simple use-case) ```classegPreprocess -d <any integer id number <= 3 digits> -dd <text description> -f <number of folds> -ext mnist_class```
6) Train the dataset ```classegTrain -d <dataset id> -f <fold to train> -ext mnist_class -m efficientnetb0_one_channel -n <experiment name>```

**What is created on the file system?**
1) When you ran the first command, ClasSeg created a configuration file at ```~/.classegrc```
2) When you preprocessed: "<pwd>/MNIST" was temporarily created, and then deleted by the proprocessor. This is where the data was downloaded to.
3) When you preprocessed: ```RAW_ROOT/Dataset_<desc>_<id>```, ```PREPROCESSED/Dataset_<desc>_<id>```. Take a look at what information is avaialble, and the default config files.
4) When you trained: ```RESULTS_ROOT/Dataset_<desc>_<id>/fold_<fold>/<experiment name>```. This is where you can find logs, **tensorboard logs**, weights, and some backups.

**A note on the environment variables**

These are used to keep the codebase clean and to allow for easy switching between datasets. If you have trouble setting them up, **you can modify classeg.utils.utils.constants.py**


# Notable Out of the Box Features
1) Multiple mode training with no coding needed. Mode is determined by file system structure.
2) Model customization and design with no coding (though it can be desireable to code for more complex systems) thanks to https://github.com/aheschl1/JsonTorchModels
   1) This is a JSON based model definition system. It is very powerful for quick iterations and modifications during model experimentation.
   2) May not be desireable all the time - create an extension and override get_model() to use a custom model.
3) Tensorboard logging
4) Extension capacity for extending functionality, while not modifying the core codebase.
   1) One codebase for multiple tasks and flows
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
Extensions currently allow you to easily create a custom trainer, inferer, and preprocessor. More easy extensionility coming soon! (you can always modify the core codebase)
1) Run **classegCreateExtension**
2) Follow the prompts
3) It is now created at ```<repo_root>/extensions/<extension_name>```
   
To use your new extension, pass the name you chose to the -extension/-ext argument for trianing, preprocessing, and inference.

**Change Trainer/Preprocessor/Inferer Class Names**

There are default names for the template classes, of course. To rename them, you need to modify **<extension_root>/__init__.py**. Modify TRAINER_CLASS_NAME, PREPROCESSOR_CLASS_NAME, and INFERER_CLASS_NAME as needed.

**Taking in custom arguments from the command line**

All trainers/preprocessors/inferers take kwargs. At the entrypoint, any argument in the form of ```<arg>=<value>``` are passed as you would expect.
In your extension classes, you can unpack them from the kwargs in __init__.

# Running Training
To run training, you need to run the following command:
```classegTrain -d <dataset id> -f <fold to train> -ext <extension name> -m <model name> -n <experiment name>```
To get more info, run ```classegTrain --help```

# Running Preprocessing
To run preprocessing, you need to run the following command:
```classegPreprocess -d <dataset id> -dd <dataset description> -f <number of folds> -ext <extension name>```.
Run ```classegPreprocess --help``` for more info.

# Running Inference
This portion is going to face a lot of changes in the future. For now, you can run inference with the following command:
```classegInfer -d <dataset id> -f <fold to infer> -ext <extension name> -i <input_folder>``` What happens if you are SSL and have no input? Too bad. Put something random for -i.
Run ```classegInfer --help``` for more details, and up to date information.