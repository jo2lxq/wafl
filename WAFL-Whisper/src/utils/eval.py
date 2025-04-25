import gc
import os

import evaluate
import torch
from tqdm import tqdm


def eval_cer_of_all_node(device, net, test_data_loader, directory, cer_result_list, total_cer_average_list, contact=None):
    metrics_cer = evaluate.load("cer")
    # Evaluate for all nodes
    for node_idx in range(len(net)):
        # Reuse previous results if there are no adjacent nodes in the collaborative learning phase
        if contact is not None:
            nbr = contact[str(node_idx)]
            if len(nbr) == 0 and (len(cer_result_list[node_idx]) != 0):
                cer_result_list[node_idx].append(cer_result_list[node_idx][-1])
                continue
        
        # Evaluation for each model
        net[node_idx].to(device)
        net[node_idx].eval()
        step = 0
        cer_sum = 0
        pred_text_all = []
        label_text_all = []

        # Batch loop with progress bar
        with torch.no_grad():
            for input_ids, labels, dec_input_ids, script, audio in tqdm(test_data_loader, desc=f"Evaluating Node {node_idx}"):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                dec_input_ids = dec_input_ids.to(device)
                label_str = []
                step += 1

                # Prediction calculation: Using model.transcribe function for production, so val_loss and val_acc are not calculated.
                pred_str = []

                for audio_path in audio:
                    result = net[node_idx].transcribe(audio_path, language="ja", task="transcribe")
                    pred_str.append(result["text"])
                    pred_text_all.append(result["text"])

                # Load scripts as reference
                for script_path in script:
                    with open(script_path) as f:
                        script_text = f.read()
                        label_str.append(script_text)
                        label_text_all.append(script_text)

                print(result["text"], script_text)

                # Calculate CER
                cer = 100 * metrics_cer.compute(predictions=pred_str, references=label_str)
                cer_sum += cer
                # Calculate average of CER and WER.
                avg_cer = cer_sum / step

                # Display CER in progress bar
                tqdm(test_data_loader, desc=f"Evaluating Node {node_idx}").set_postfix(cer=avg_cer)

        avg_cer = cer_sum / (step)
        cer_result_list[node_idx].append(avg_cer)

        sorted_idx = sorted(range(len(label_text_all)), key=lambda x: label_text_all[x])
        sorted_pred_text_all = [pred_text_all[i] for i in sorted_idx]
        sorted_label_text_all = [label_text_all[i] for i in sorted_idx]

        for i in range(len(pred_text_all)):
            if not os.path.exists(f"{directory}/text/node{node_idx}"):
                print("make_dir")
                os.makedirs(f"{directory}/text/node{node_idx}", exist_ok=True)
            with open(f"{directory}/text/node{node_idx}/{i}.txt", "a") as f:
                f.write(f"pred:{sorted_pred_text_all[i]} label:{sorted_label_text_all[i]}\n")

        net[node_idx].to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate average CER for all nodes
    if total_cer_average_list is not None:
        total_cer_average_list.append(sum([cer_result_list[i][-1] for i in range(len(net))]) / len(net))
    
    return cer_result_list, total_cer_average_list
