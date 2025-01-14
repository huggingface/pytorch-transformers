import copy
import json
import os
import tempfile
import logging
from typing import List, Type, Optional, Any

from transformers import is_torch_available

from .utils.test_configuration_utils import config_common_kwargs


class ConfigTester:
    def __init__(
        self,
        parent: Any,
        config_class: Optional[Type] = None,
        has_text_modality: bool = True,
        common_properties: Optional[List[str]] = None,
        **kwargs: Any
    ):
        self.parent = parent
        self.config_class = config_class
        self.has_text_modality = has_text_modality
        self.inputs_dict = kwargs
        self.common_properties = common_properties or [
            "hidden_size", "num_attention_heads", "num_hidden_layers"
        ]
        if self.has_text_modality:
            self.common_properties.append("vocab_size")

    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        
        for prop in self.common_properties:
            if not hasattr(config, prop):
                self.parent.fail(f"`{prop}` does not exist in the config class.")
                
            try:
                setattr(config, prop, 1)
                self.parent.assertEqual(
                    getattr(config, prop), 1, 
                    msg=f"`{prop}` expected to be 1, but got {getattr(config, prop)}"
                )
            except NotImplementedError:
                logging.warning(f"Setter for `{prop}` not implemented.")

    def create_and_test_config_to_json_string(self):
        config = self.config_class(**self.inputs_dict)
        obj = json.loads(config.to_json_string())
        
        for key, value in self.inputs_dict.items():
            self.parent.assertEqual(
                obj.get(key), value, 
                msg=f"`{key}` expected to be {value}, but got {obj.get(key)}"
            )

    def create_and_test_config_to_json_file(self):
        config_first = self.config_class(**self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "config.json")
            config_first.to_json_file(json_file_path)
            config_second = self.config_class.from_json_file(json_file_path)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def create_and_test_config_from_and_save_pretrained(self):
        config_first = self.config_class(**self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config_first.save_pretrained(tmpdirname)
            config_second = self.config_class.from_pretrained(tmpdirname)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

        with self.parent.assertRaises(OSError):
            invalid_path = f".{tmpdirname}"
            logging.info(f"Attempting to load from invalid path: {invalid_path}")
            self.config_class.from_pretrained(invalid_path)

    def create_and_test_config_from_and_save_pretrained_subfolder(self):
        config_first = self.config_class(**self.inputs_dict)

        subfolder = "test"
        with tempfile.TemporaryDirectory() as tmpdirname:
            sub_tmpdirname = os.path.join(tmpdirname, subfolder)
            config_first.save_pretrained(sub_tmpdirname)
            config_second = self.config_class.from_pretrained(tmpdirname, subfolder=subfolder)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def create_and_test_config_with_num_labels(self):
        config = self.config_class(**self.inputs_dict, num_labels=5)
        
        self.parent.assertEqual(len(config.id2label), 5)
        self.parent.assertEqual(len(config.label2id), 5)

        config.num_labels = 3
        self.parent.assertEqual(len(config.id2label), 3)
        self.parent.assertEqual(len(config.label2id), 3)

    def check_config_can_be_init_without_params(self):
        if self.config_class.is_composition:
            with self.parent.assertRaises(ValueError):
                self.config_class()
        else:
            config = self.config_class()
            self.parent.assertIsNotNone(config)

    def check_config_arguments_init(self):
        kwargs = copy.deepcopy(config_common_kwargs)
        config = self.config_class(**kwargs)
        wrong_values = []

        if "torch_dtype" in kwargs and is_torch_available():
            import torch
            if config.torch_dtype != torch.float16:
                wrong_values.append(("torch_dtype", config.torch_dtype, torch.float16))
        
        for key, value in kwargs.items():
            if key != "torch_dtype" and getattr(config, key) != value:
                wrong_values.append((key, getattr(config, key), value))

        if wrong_values:
            errors = "\n".join([f"- {v[0]}: got {v[1]} instead of {v[2]}" for v in wrong_values])
            raise ValueError(f"The following keys were not properly set in the config:\n{errors}")

    def run_common_tests(self):
        self.create_and_test_config_common_properties()
        self.create_and_test_config_to_json_string()
        self.create_and_test_config_to_json_file()
        self.create_and_test_config_from_and_save_pretrained()
        self.create_and_test_config_from_and_save_pretrained_subfolder()
        self.create_and_test_config_with_num_labels()
        self.check_config_can_be_init_without_params()
        self.check_config_arguments_init()
