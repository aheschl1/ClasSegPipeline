from setuptools import find_packages
from setuptools import setup

setup(
    name='classeg',
    version='0.2.0',
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "torchvision",
        "pillow",
        "overrides",
        "click",
        "scikit-learn",
        "seaborn",
        "pyyaml",
        "tqdm",
        "seaborn",
        "pandas",
        "SimpleITK",
        "einops",
        "albumentations",
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
