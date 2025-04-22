import gc
import os

import evaluate
import torch
from tqdm import tqdm


def eval_cer_of_each_node(device, n, net, test_data_loader, directory, cer_result_list, cer_average_list):
    metrics_cer = evaluate.load("cer")

    # 各モデルでの評価
    net[n].to(device)
    net[n].eval()
    step = 0
    cer_sum = 0
    pred_text_all = []
    label_text_all = []

    # バッチのループ、プログレスバー対応
    with torch.no_grad():
        for input_ids, labels, dec_input_ids, script, audio in tqdm(test_data_loader, desc="val"):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            dec_input_ids = dec_input_ids.to(device)
            label_str = []
            step += 1

            # 予測計算　本番用の model.transcribe 関数を用いて予測するため、val_loss と val_acc は計算しない。
            pred_str = []

            for audio_path in audio:
                result = net[n].transcribe(audio_path, language="ja", task="transcribe")
                pred_str.append(result["text"])
                pred_text_all.append(result["text"])

            # Load scripts as reference
            for script_path in script:
                with open(script_path) as f:
                    script_text = f.read()
                    label_str.append(script_text)
                    label_text_all.append(script_text)

            print(result["text"], script_text)

            # cer 算出
            cer = 100 * metrics_cer.compute(predictions=pred_str, references=label_str)
            cer_sum += cer
            # cer と wer の平均値計算。
            avg_cer = cer_sum / step

            # プログレスバーに cer 表示
            tqdm(test_data_loader, desc="val").set_postfix(cer=avg_cer)

    avg_cer = cer_sum / (step)
    cer_result_list[n].append(avg_cer)

    sorted_idx = sorted(range(len(label_text_all)), key=lambda x: label_text_all[x])
    sorted_pred_text_all = [pred_text_all[i] for i in sorted_idx]
    sorted_label_text_all = [label_text_all[i] for i in sorted_idx]

    cer_average_list.append(sum([cer_result_list[i][-1] for i in range(len(net))]) / len(net))

    for i in range(len(pred_text_all)):
        if not os.path.exists(f"{directory}/text/node{n}/native"):
            print("make_dir")
            os.makedirs(f"{directory}/text/node{n}/native", exist_ok=True)
        with open(f"{directory}/text/node{n}//native/{i}.txt", "a") as f:
            f.write(f"pred:{sorted_pred_text_all[i]} label:{sorted_label_text_all[i]}\n")

    net[n].to("cpu")
    gc.collect()  # ←―――――――――――――――――――
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cer_result_list, cer_average_list
