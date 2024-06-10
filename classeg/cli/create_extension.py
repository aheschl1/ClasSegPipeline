import os
import shutil

from classeg.utils.constants import SEGMENTATION, CLASSIFICATION, SELF_SUPERVISED, EXTENSIONS_DIR

# ../../extensions
# classeg.extensions
TEMPLATE_TRAINING = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'extensions/*/training/trainer.py')
TEMPLATE_INFERENCE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'extensions/*/inference/inferer.py')
TEMPLATE_PREPROCESSING = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'extensions/*/preprocessing/preprocessor.py')

TEMPLATE_DATAPOINT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataloading/templates/datapoint.py')
TEMPLATE_DATASET = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataloading/templates/dataset.py')


def copy_templates(template_type, root_path):
    assert template_type in [SEGMENTATION, CLASSIFICATION, SELF_SUPERVISED], f"Invalid template type {template_type}"
    star_replacement = {
        SEGMENTATION: "default_seg",
        CLASSIFICATION: "default_class",
        SELF_SUPERVISED: "default_ssl"
    }[template_type]
    shutil.copy(TEMPLATE_TRAINING.replace("*", star_replacement), f"{root_path}/training/trainer.py")
    shutil.copy(TEMPLATE_PREPROCESSING.replace("*", star_replacement), f"{root_path}/preprocessing")
    shutil.copy(TEMPLATE_INFERENCE.replace("*", star_replacement), f"{root_path}/inference/inferer.py")


def create_extension(name, template_type, custom_dataset=False, custom_datapoint=False):
    # Define the path for the new extension
    extension_path = str(os.path.join(EXTENSIONS_DIR, name))
    trainer_class_name = {
        CLASSIFICATION: "ClassificationTrainer",
        SELF_SUPERVISED: "SelfSupervisedTrainer",
        SEGMENTATION: "SegmentationTrainer"
    }[template_type]

    inferer_class_name = {
        CLASSIFICATION: "ClassificationInferer",
        SELF_SUPERVISED: "SelfSupervisedInferer",
        SEGMENTATION: "SegmentationInferer"
    }[template_type]

    if not os.path.exists(extension_path):
        os.makedirs(extension_path)
        with open(os.path.join(extension_path, '__init__.py'), 'w') as f:
            f.write(f'# {name} extension. Pass -ext {name} to the entrypoints to use.\n\n')
            f.write("# To override the Datapoint and PipelineDataset used in the extension, you can create a new file\n"
                    "# Create ..dataloading.dataset.Datapoint and ..dataloading.dataset.PipelineDataset\n"
                    "# This will make the global state access your custom classes as supposed to the default ones\n"
                    "# If you make a new dataset and datapoint, you may need to do some debugging if you change construtors and return schemes and such\n"
                    "# It is recommended that you continue extending PipelineDataset and Datapoint regardless.\n\n")
            f.write(f'TRAINER_CLASS_NAME = "{trainer_class_name}"\n'
                    f'PREPROCESSOR_CLASS_NAME = "ExtensionPreprocessor"\n'
                    f'INFERER_CLASS_NAME = "{inferer_class_name}"\n\n')
    else:
        raise SystemExit(f"Extension '{name}' already exists.")

    trainer_path = f"{extension_path}/training"
    preprocess_path = f"{extension_path}/preprocessing"
    inferer_path = f"{extension_path}/inference"

    os.mkdir(trainer_path)
    os.mkdir(preprocess_path)
    os.mkdir(inferer_path)

    open(os.path.join(f"{extension_path}/training", '__init__.py'), "w+")
    open(os.path.join(f"{extension_path}/preprocessing", '__init__.py'), "w+")
    open(os.path.join(f"{extension_path}/inference", '__init__.py'), "w+")
    copy_templates(template_type, extension_path)

    if custom_dataset or custom_datapoint:
        os.mkdir(f"{extension_path}/dataloading")
        open(f"{extension_path}/dataloading/__init__.py", "w+")
        if custom_datapoint:
            shutil.copy(f"{TEMPLATE_DATAPOINT}", f"{extension_path}/dataloading/datapoint.py")
        if custom_dataset:
            shutil.copy(f"{TEMPLATE_DATASET}", f"{extension_path}/dataloading/dataset.py")

    return extension_path


def extension_exists(name):
    if os.path.exists(os.path.join(EXTENSIONS_DIR, name)):
        print("Extension exists")
        return True
    return False


def main():
    name = None
    while name is None or '-' in name or '.' in name:
        while name is None or extension_exists(name):
            name = input('Enter extension name: ')

    extension_type = None
    while extension_type is None:
        type_query = "What extension template? 1) Classification 2) Segmentation 3) Self Supervised: "
        extension_type = {
            1: CLASSIFICATION,
            2: SEGMENTATION,
            3: SELF_SUPERVISED
        }.get(int(input(type_query)), None)

    print("============Info============")
    print("If you choose to create a custom dataset or datapoint, consider that you will not receive updates when you sync your fork!")
    print("============================")
    custom_dataset = input("Do you want to design a custom dataset? (y/n): ").strip().lower() in ["y", "1", "yes"]
    custom_datapoint = input("Do you want to design a custom datapoint? (y/n): ").strip().lower() in ["y", "1", "yes"]

    ext_dir = create_extension(name, extension_type, custom_dataset=custom_dataset, custom_datapoint=custom_datapoint)
    print(f"Extension created at {ext_dir}")


if __name__ == '__main__':
    main()
