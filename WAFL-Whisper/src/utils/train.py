import gc

import torch
from tqdm import tqdm


def train_each_model(device, net, n, train_loader_list, optimizer, criterion):
    net[n].to(device)
    net[n].train()

    # Training loop with progress bar
    pbar = tqdm(train_loader_list[n], desc="Training")
    for input_ids, labels, dec_input_ids, _, _ in pbar:
        input_ids, labels, dec_input_ids = (
            input_ids.to(device),
            labels.to(device),
            dec_input_ids.to(device),
        )

        optimizer[n].zero_grad()
        logits = net[n](input_ids, tokens=dec_input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer[n].step()

        pbar.set_postfix(loss=loss.item())

    # 学習直後にいったんGC
    net[n].to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
