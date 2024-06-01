import os
import shutil
import sys
from pipe.utils.constants import SEGMENTATION, CLASSIFICATION, SELF_SUPERVISED, EXTENSIONS_DIR

# ../../extensions
# pipe.extensions
TEMPLATES_TRAINING = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training/default_trainers')
TEMPLATE_PREPROCESSING = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocessing/templates/preprocessor.py')


def copy_templates(template_type, root_path):
    assert template_type in [SEGMENTATION, CLASSIFICATION, SELF_SUPERVISED], "Invalid template type"
    shutil.copy(f"{TEMPLATES_TRAINING}/{template_type}_trainer.py", f"{root_path}/training/trainer.py")
    shutil.copy(TEMPLATE_PREPROCESSING, f"{root_path}/preprocessing")


def create_extension(name, template_type):
    # Define the path for the new extension
    extension_path = os.path.join(EXTENSIONS_DIR, name)

    if not os.path.exists(extension_path):
        os.makedirs(extension_path)
        with open(os.path.join(extension_path, '__init__.py'), 'w') as f:
            f.write(f'# {name} extension\n')
    else:
        print(f"Extension '{name}' already exists.")

    trainer_path = f"{extension_path}/training"
    preprocess_path = f"{extension_path}/preprocessing"
    os.mkdir(trainer_path)
    os.mkdir(preprocess_path)
    open(os.path.join(f"{extension_path}/training", '__init__.py'), "w+")
    open(os.path.join(f"{extension_path}/preprocessing", '__init__.py'), "w+")
    copy_templates(template_type, extension_path)

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
    ext_dir = create_extension(name, extension_type)
    print(f"Extension created at {ext_dir}")


if __name__ == '__main__':
    main()
