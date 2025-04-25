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
            
    def get_contact_list(self):
        """
        通信パターンファイルからcontact_listを読み込む
        
        Returns:
            list: 各エポックでの通信パターンを格納したリスト
        """
        with open(self.contact_file) as f:
            contact_list = json.load(f)
        return contact_list
        
    def save_results(self, cer_result_of_each_node, total_cer_average_list):
        """
        実験結果を保存し、グラフを描画する
        
        Args:
            cer_result_of_each_node: 各ノードのCER結果
            total_cer_average_list: エポックごとの平均CER
        """
        # 結果の保存、グラフの描画
        with open(f"{self.output_dir}/all_result.txt", "w") as f:
            f.write("=== 実験設定 ===\n")
            f.write(f"実験メモ: {self.memo}\n")
            f.write(f"ノード数: {self.n_node}\n")
            f.write(f"事前学習エポック数: {self.pre_epoch}\n")
            f.write(f"協調学習エポック数: {self.num_epoch}\n")
            f.write(f"学習率: {self.lr}\n")
            f.write(f"学習用バッチサイズ: {self.train_batch_size}\n")
            f.write(f"テスト用バッチサイズ: {self.test_batch_size}\n")
            f.write(f"通信パターンファイル: {self.contact_file}\n")
            f.write(f"データセット: {self.data_name_list}\n")
            f.write(f"データディレクトリ: {self.data_dir}\n")
            f.write(f"連合学習係数: {self.fl_coefficiency}\n")
            f.write(f"乱数シード: {self.seed}\n")
            f.write(f"出力ディレクトリ: {self.output_dir}\n")
            f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n=== 結果 ===\n")
            f.write(f"最終平均CER: {total_cer_average_list[-1]}\n")

        # CER結果をJSON形式で保存
        cer_results = {
            "node_results": {f"node_{node}": cer_result_of_each_node[node] for node in range(self.n_node)},
            "average_results": total_cer_average_list,
        }
        with open(f"{self.output_dir}/cer_results.json", "w", encoding="utf-8") as f:
            json.dump(cer_results, f, indent=2, ensure_ascii=False)

        # グラフの描画
        plt.plot(total_cer_average_list)
        plt.xlabel("epoch")
        plt.ylabel("CER")
        plt.title("Average CER of All Nodes")
        plt.savefig(f"{self.output_dir}/graph/average_cer.png")
        plt.close()
