import torch
import torch.nn as nn
import whisper

from src.config.config_manager import ConfigManager
from src.data.dataset import prepare_datasets, prepare_test_dataset
from src.model.exchange import exchange_parameter_with_close_nodes, save_model
from src.utils.eval import eval_cer_of_all_node
from src.utils.train import train_all_model


def main(config_path="config.json"):
    # Initialize configuration
    config = ConfigManager(config_path)
    device = config.get_device()
    config.setup_output_directory()

    # Model configuration
    woptions = whisper.DecodingOptions(language="ja", without_timestamps=True)
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language="ja", task=woptions.task)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    net = [whisper.load_model(config.whisper_model_size, device="cpu") for _ in range(config.n_node)]

    # Optimizer configuration
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

    contact_list = config.get_contact_list()

    cer_result_of_each_node = [[] for _ in range(config.n_node)]
    total_cer_average_list = []
    local_model_parameter = [{} for _ in range(config.n_node)]

    # Evaluate pretrained model
    cer_result_of_each_node, total_cer_average_list = eval_cer_of_all_node(
        device,
        net,
        test_data_loader,
        config.output_dir,
        cer_result_of_each_node,
        total_cer_average_list,
    )

    # Pre-training phase
    for epoch in range(config.pre_epoch):
        train_all_model(device, net, train_loader_list, optimizer, criterion)
        
        cer_result_of_each_node, total_cer_average_list = eval_cer_of_all_node(
            device,
            net,
            test_data_loader,
            config.output_dir,
            cer_result_of_each_node,
            total_cer_average_list,
        )

    # Save model after pre-training epochs
    save_model(net, config.output_dir, "preepoch")

    # Collaborative learning phase
    for epoch in range(config.num_epoch):
        contact = contact_list[epoch]
        print(f"at t={epoch} : ", contact)

        # TODO Want to consolidate loading into a single function
        local_model_parameter = exchange_parameter_with_close_nodes(
            net, contact, config.n_node, config.fl_coefficiency
        )

        for n in range(config.n_node):
            nbr = contact[str(n)]
            if len(nbr) > 0:
                net[n].load_state_dict(local_model_parameter[n])

        # Training for each model
        train_all_model(device, net, train_loader_list, optimizer, criterion, contact)

        # Evaluation for each model
        cer_result_of_each_node, total_cer_average_list = eval_cer_of_all_node(
            device,
            net,
            test_data_loader,
            config.output_dir,
            cer_result_of_each_node,
            total_cer_average_list,
            contact,
        )

    # Save model and results
    save_model(net, config.output_dir, "final_epoch")
    
    # Save results and draw graphs
    config.save_results(cer_result_of_each_node, total_cer_average_list)


if __name__ == "__main__":
    main(config_path="config.json")
