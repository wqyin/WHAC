import os
import importlib.util
from pathlib import Path
import json


class Config(dict):
    """A dictionary that allows dot notation access for configuration settings."""
    def __init__(self, data=None):
        super().__init__()
        if data:
            for key, value in data.items():
                # Set the key-value pair using the key and the converted value
                self[key] = self._convert(value)

    def _convert(self, value):
        """Recursively convert nested dictionaries to Config."""
        if isinstance(value, dict):
            return Config(value)  # Convert all nested dicts to Config
        elif isinstance(value, list):
            return [self._convert(item) for item in value]  # Convert items in lists
        elif isinstance(value, Path):
            return str(value)  # Convert Path objects to string
        return value

    def __getattr__(self, item):
        """Allow access to dictionary keys via dot notation."""
        if item in self:
            return self[item]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
    
    def __setattr__(self, key, value):
        """Allow setting dictionary keys via dot notation."""
        self[key] = self._convert(value)
    
    @classmethod
    def load_config(cls, file_path):
        """Load a Python config file and return it as a Config instance."""
        spec = importlib.util.spec_from_file_location("config_module", file_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        if hasattr(config_module, "config") and isinstance(config_module.config, dict):
            return cls(config_module.config)  # Ensure full conversion of nested dicts
        else:
            raise ValueError("The config file does not define a 'config' dictionary.")

    def update_config(self, new_data):
        """Recursively update Config with new dictionary values."""
        for key, value in new_data.items():
            if isinstance(value, dict) and isinstance(self.get(key), Config):
                self[key].update_config(value)  # Recursive update for nested dicts
            else:
                self[key] = self._convert(value)  # Convert and assign
    
    def dump_config(self, file_path=None):
        """Dump the Config object into a .py file with a Pythonic and readable format."""
        # Ensure the provided path is valid
        if file_path is None:
            file_path = self.log.output_dir + '/config.py'
        else:
            dir_name = os.path.dirname(file_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)

        # Convert the Config instance into a regular dictionary
        def config_to_dict(config):
            """Recursively convert a Config instance into a regular dictionary."""
            if isinstance(config, Config):
                return {key: config_to_dict(value) if isinstance(value, Config) else value
                        for key, value in config.items()}
            return config

        config_dict = config_to_dict(self)

        # Write the config dictionary to a .py file in a formatted, readable way
        with open(file_path, 'w') as f:
            # Use json.dumps to pretty-print the dictionary with indentation and spaces
            f.write("config = ")
            f.write(json.dumps(config_dict, indent=4))  # Pretty print with indentation
            f.write("\n")

        print(f"Config has been saved to {file_path}")

    def prepare_log(self):

        def make_folder(folder):
            if not os.path.exists(folder):
                os.makedirs(folder)

        if self.log.output_dir is not None:
            make_folder(self.log.output_dir) 
        if self.log.model_dir is not None:
            make_folder(self.log.model_dir) 
        if self.log.log_dir is not None:
            make_folder(self.log.log_dir) 
        if self.log.result_dir is not None:
            make_folder(self.log.result_dir) 



