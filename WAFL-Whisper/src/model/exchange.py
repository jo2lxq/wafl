import torch


def exchange_parameter_with_close_nodes(net, contact, n_node, fl_coefficiency):
    local_model = [{} for _ in range(n_node)]
    recv_models = [[] for _ in range(n_node)]
    update_model = [{} for _ in range(n_node)]

    # Receive local_models from contacts
    for n in range(n_node):
        local_model[n] = net[n].state_dict()
        nbr = contact[str(n)]  # Models that the nth model can communicate with
        recv_models[n] = []
        for k in nbr:
            recv_models[n].append(net[k].state_dict())  # Array of parameters of models communicating with the nth model

    for n in range(n_node):
        update_model = recv_models[n]
        n_nbr = len(update_model)
        print(f"at {n} n_nbr={n_nbr}")

        # Calculate the difference between own and other models
        for k in range(n_nbr):
            for key in update_model[k]:
                update_model[k][key] = recv_models[n][k][key] - local_model[n][key]

        # Update parameters using the differences
        for k in range(n_nbr):
            for key in update_model[k]:
                local_model[n][key] += update_model[k][key] * fl_coefficiency / (n_nbr + 1)

    return local_model


def save_model(net, output_dir, epoch_type):
    for n in range(len(net)):
        save_path = f"{output_dir}/model/node{n}_{epoch_type}.pth"
        torch.save(net[n].state_dict(), save_path)
        print(f"Node {n} params saved to {save_path}")
