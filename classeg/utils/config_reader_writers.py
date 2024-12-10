import json
import yaml
import os


class ConfigReader:
    def __init__(self, root, name):
        self.root = root
        self.name = name

    def read(self):
        """
        Read a json or yaml file from the root directory.
        First checks if the file is a json file, then a yaml file.
        Returns the contents of the file.
        -------

        """
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"File {self.root} does not exist.")
        if os.path.exists(f"{self.root}/{self.name}.json"):
            with open(f"{self.root}/{self.name}.json", 'r') as f:
                return json.load(f)
        elif os.path.exists(f"{self.root}/{self.name}.yaml"):
            with open(f"{self.root}/{self.name}.yaml", 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"File {self.root}/{self.name} is not a valid json or yaml file.")

    @staticmethod
    def read_from_root(root, name):
        """
        Read a json or yaml file from the root directory.
        First checks if the file is a json file, then a yaml file.
        Returns the contents of the file.
        -------

        """
        if not os.path.exists(root):
            raise FileNotFoundError(f"File {root} does not exist.")
        if os.path.exists(f"{root}/{name}.json"):
            with open(f"{root}/{name}.json", 'r') as f:
                return json.load(f)
        elif os.path.exists(f"{root}/{name}.yaml"):
            with open(f"{root}/{name}.yaml", 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"File {root}/{name} is not a valid json or yaml file.")
