import json
import os
import random
import re
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch


class ConfigManager:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self._load_config()
        self._set_seed(self.seed)
        self._setup_environment()

    def _load_config(self):
        with open(self.config_path, "r") as f:
            config = json.load(f)
        self.memo = config["memo"]
        self.n_node = config["n_node"]
        self.pre_epoch = config["pre_epoch"]
        self.num_epoch = config["num_epoch"]
        self.lr = config["lr"]
        self.contact_file = config["contact_file"]
        self.data_name_list = config["data_name_list"]
        self.data_dir = config["data_dir"]
        self.fl_coefficiency = config["fl_coefficiency"]
        self.seed = config["seed"]
        self.output_dir = config["output_dir"]
        self.train_batch_size = config["train_batch_size"]
        self.test_batch_size = config["test_batch_size"]

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _setup_environment(self):
        warnings.simplefilter("ignore")
        plt.rcParams["font.size"] = 14
        plt.rcParams["figure.figsize"] = (6, 6)
        plt.rcParams["axes.grid"] = True
        np.set_printoptions(suppress=True, precision=5)

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def setup_output_directory(self):
        match = re.search(r"(rwp_[^/]+)\.json", self.contact_file)
        if match:
            contact_file_name = match.group(1)
            print(contact_file_name)
        else:
            print(r"not match expected contact_file: '(rwp_[^/]+)\.json'")
            raise ValueError("contact_file_name is not found")

        self.output_dir = (
            f"{self.output_dir}/{datetime.now().strftime('%Y%m%d-%H%M%S')}_{self.memo}_{self.config_path}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/graph", exist_ok=True)
        os.makedirs(f"{self.output_dir}/text", exist_ok=True)
        os.makedirs(f"{self.output_dir}/model", exist_ok=True)
        for n in range(self.n_node):
            os.makedirs(f"{self.output_dir}/text/node{n}", exist_ok=True)
