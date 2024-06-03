from setuptools import find_packages
from setuptools import setup

setup(
    name='classeg',
    version='0.2.0',
    install_requires=[
        "torch",
        "numpy~=1.26.3",
        "matplotlib~=3.8.0",
        "torchvision~=0.17.0",
        "pillow~=10.2.0",
        "overrides~=7.4.0",
        "click~=8.1.7",
        "scikit-learn~=1.5.0",
        "seaborn",
        "pyyaml~=6.0.1",
        "tqdm~=4.65.0",
        "seaborn~=0.12.2",
        "pandas~=2.1.4",
        "SimpleITK~=2.3.1",
        "einops~=0.7.0",
        "albumentations~=1.4.8",
        "opencv-python",
        "tensorboard",
        "json-torch-models",
        "monai",
        "multiprocessing-logging",
        "tensorboard",
    ],
    entry_points={
        'console_scripts': [
            "classegPreprocess=classeg.cli.preprocess_entry:main",
            "classegTrain=classeg.cli.training_entry:main",
            "classegCreateExtension=classeg.cli.create_extension:main",
            "classegInference=classeg.cli.inference_entry:main"
        ]
    },
    packages=find_packages(),
    url='https://github.com/aheschl1/ClasSegPipeline',
    author='Andrew Heschl',
    author_email='andrew.heschl@ucalgary.ca',
    description='Flexible Deep Learning Package'
)