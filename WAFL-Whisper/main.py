import json
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import whisper

from src.config.config_manager import ConfigManager
from src.data.dataset import prepare_datasets, prepare_test_dataset
from src.model.exchange import exchange_parameter_with_close_nodes, save_model
from src.utils.eval import eval_cer_of_all_node
from src.utils.train import train_each_model


def main(config_path="config.json"):
    # 設定の初期化
    config = ConfigManager(config_path)
    device = config.get_device()
    config.setup_output_directory()

    # モデルの設定
    woptions = whisper.DecodingOptions(language="ja", without_timestamps=True)
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language="ja", task=woptions.task)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    net = [whisper.load_model("tiny", device="cpu") for _ in range(config.n_node)]

    # オプティマイザの設定
    optimizer = [
        torch.optim.SGD(
            [param for param in net[i].parameters() if param.requires_grad],
            lr=config.lr,
            momentum=0.9,
        )
        for i in range(config.n_node)
    ]

    train_loader_list = prepare_datasets(config.data_name_list, config.data_dir, wtokenizer, config.train_batch_size)
    test_data_loader = prepare_test_dataset(config.data_dir, wtokenizer, config.test_batch_size)

    with open(config.contact_file) as f:
        contact_list = json.load(f)

    cer_result_of_each_node = [[] for _ in range(config.n_node)]
    total_cer_average_list = []
    local_model_parameter = [{} for _ in range(config.n_node)]

    # pretrained modelの評価
    cer_result_of_each_node, total_cer_average_list = eval_cer_of_all_node(
        device,
        net,
        test_data_loader,
        config.output_dir,
        cer_result_of_each_node,
        total_cer_average_list,
    )

    # 事前学習フェーズ
    for epoch in range(config.pre_epoch):
        for n in range(config.n_node):
            train_each_model(device, net, n, train_loader_list, optimizer, criterion)
        
        cer_result_of_each_node, total_cer_average_list = eval_cer_of_all_node(
            device,
            net,
            test_data_loader,
            config.output_dir,
            cer_result_of_each_node,
            total_cer_average_list,
        )

    # pre epoch後のモデルを保存
    save_model(net, config.output_dir, "preepoch")

    # 協調学習フェーズ
    for epoch in range(config.num_epoch):
        contact = contact_list[epoch]
        print(f"at t={epoch} : ", contact)

        # TODO ロードもまとめて一つの関数にしたい
        local_model_parameter = exchange_parameter_with_close_nodes(
            net, contact, config.n_node, config.fl_coefficiency
        )

        for n in range(config.n_node):
            nbr = contact[str(n)]
            if len(nbr) > 0:
                net[n].load_state_dict(local_model_parameter[n])

        # 各モデルでの学習(Training)
        for n in range(config.n_node):
            nbr = contact[str(n)]
            if len(nbr) == 0:
                print(f"Node {n} has no neighbor")
            else:
                train_each_model(device, net, n, train_loader_list, optimizer, criterion)

        # 各モデルでの評価(Evaluation)
        cer_result_of_each_node, total_cer_average_list = eval_cer_of_all_node(
            device,
            net,
            test_data_loader,
            config.output_dir,
            cer_result_of_each_node,
            total_cer_average_list,
            contact,
        )

    # モデルと結果の保存
    save_model(net, config.output_dir, "final_epoch")

    # 結果の保存、グラフの描画
    with open(f"{config.output_dir}/all_result.txt", "w") as f:
        f.write("=== 実験設定 ===\n")
        f.write(f"実験メモ: {config.memo}\n")
        f.write(f"ノード数: {config.n_node}\n")
        f.write(f"事前学習エポック数: {config.pre_epoch}\n")
        f.write(f"協調学習エポック数: {config.num_epoch}\n")
        f.write(f"学習率: {config.lr}\n")
        f.write(f"学習用バッチサイズ: {config.train_batch_size}\n")
        f.write(f"テスト用バッチサイズ: {config.test_batch_size}\n")
        f.write(f"通信パターンファイル: {config.contact_file}\n")
        f.write(f"データセット: {config.data_name_list}\n")
        f.write(f"データディレクトリ: {config.data_dir}\n")
        f.write(f"連合学習係数: {config.fl_coefficiency}\n")
        f.write(f"乱数シード: {config.seed}\n")
        f.write(f"出力ディレクトリ: {config.output_dir}\n")
        f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n=== 結果 ===\n")
        f.write(f"最終平均CER: {total_cer_average_list[-1]}\n")

    # CER結果をJSON形式で保存
    cer_results = {
        "node_results": {f"node_{node}": cer_result_of_each_node[node] for node in range(config.n_node)},
        "average_results": total_cer_average_list,
    }
    with open(f"{config.output_dir}/cer_results.json", "w", encoding="utf-8") as f:
        json.dump(cer_results, f, indent=2, ensure_ascii=False)

    plt.plot(total_cer_average_list)
    plt.xlabel("epoch")
    plt.ylabel("CER")
    plt.title("Average CER of All Nodes")
    plt.savefig(f"{config.output_dir}/graph/average_cer.png")
    plt.close()


if __name__ == "__main__":
    main(config_path="config.json")
