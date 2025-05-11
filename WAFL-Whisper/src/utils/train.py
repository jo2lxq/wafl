import gc

import torch
from tqdm import tqdm


def train_all_model(device, net, train_loader_list, optimizer, criterion, contact=None):
    # Train for all nodes
    for node_idx in range(len(net)):
        # Skip training if there are no adjacent nodes in the collaborative learning phase
        if contact is not None:
            nbr = contact[str(node_idx)]
            if len(nbr) == 0:
                print(f"Node {node_idx} has no neighbor, skipping training")
                continue
        
        net[node_idx].to(device)
        net[node_idx].train()

        # Training loop with progress bar
        pbar = tqdm(train_loader_list[node_idx], desc=f"Training Node {node_idx}")
        for input_ids, labels, dec_input_ids, _, _ in pbar:
            input_ids, labels, dec_input_ids = (
                input_ids.to(device),
                labels.to(device),
                dec_input_ids.to(device),
            )

            optimizer[node_idx].zero_grad()
            logits = net[node_idx](input_ids, tokens=dec_input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer[node_idx].step()

            pbar.set_postfix(loss=loss.item())

        # Run garbage collection immediately after training
        net[node_idx].to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
