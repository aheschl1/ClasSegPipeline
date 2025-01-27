# llm_brain_seg extension

# This file is used to define the trainer, preprocessor and inferer class names for the extension.

# Note that you can define your custom dataset by overriding get_dataloaders() in trainer, and writing a custom
# preprocessing scheme. More centralized dataset definitions will come.

TRAINER_CLASS_NAME = "SegmentationTrainer"
PREPROCESSOR_CLASS_NAME = "ExtensionPreprocessor"
INFERER_CLASS_NAME = "SegmentationInferer"
