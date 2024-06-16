import copy

import torch


def update_nets_vgg(net, contact, fl_coefficiency):  # sはn番目の要素に複数のmodelがあるかで決まる
    local_model = [{} for _ in range(10)]
    recv_models = [[] for _ in range(10)]
    for n in range(10):
        local_model[n] = net[n].classifier[6].state_dict()
        nbr = contact[str(n)]  # the nodes n-th node contacted
        recv_models[n] = []
        for k in nbr:
            recv_models[n].append(copy.deepcopy(net[k].classifier[6].state_dict())) # 参照渡しを防ぐために変更

    # mixture of models
    for n in range(10):
        update_model = copy.deepcopy(recv_models[n])
        n_nbr = len(update_model)  # how many nodes n-th node contacted

        # put difference of n-th node models and k-th conducted node to n-th into update_model[k]
        for k in range(n_nbr):
            for key in update_model[k]:
                update_model[k][key] = recv_models[n][k][key] - local_model[n][key]

        # mix to local model
        for k in range(n_nbr):
            for key in update_model[k]:
                local_model[n][key] += (
                    update_model[k][key] * fl_coefficiency / float(n_nbr + 1)
                )
    # update nets
    for n in range(10):
        nbr = contact[str(n)]
        if len(nbr) > 0:
            net[n].classifier[6].load_state_dict(local_model[n])


def update_nets_res(net, contact, fl_coefficiency):  # sはn番目の要素に複数のmodelがあるかで決まる
    local_model = [{} for _ in range(10)]
    recv_models = [[] for _ in range(10)]
    for n in range(10):
        local_model[n] = net[n].fc.state_dict()
        nbr = contact[str(n)]  # the nodes n-th node contacted
        recv_models[n] = []
        for k in nbr:
            recv_models[n].append(copy.deepcopy(net[k].fc.state_dict())) # 参照渡しを防ぐために変更

    # mixture of models
    for n in range(10):
        update_model = copy.deepcopy(recv_models[n]) # 参照渡しを防ぐために変更
        n_nbr = len(update_model)  # how many nodes n-th node contacted

        # put difference of n-th node models and k-th conducted node to n-th into update_model[k]
        for k in range(n_nbr):
            for key in update_model[k]:
                update_model[k][key] = recv_models[n][k][key] - local_model[n][key]

        # mix to local model
        for k in range(n_nbr):
            for key in update_model[k]:
                local_model[n][key] += (
                    update_model[k][key] * fl_coefficiency / float(n_nbr + 1)
                )
    # update nets
    for n in range(10):
        nbr = contact[str(n)]
        if len(nbr) > 0:
            net[n].fc.load_state_dict(local_model[n])


def update_nets_vit(net, contact, fl_coefficiency):  # sはn番目の要素に複数のmodelがあるかで決まる
    local_model = [{} for _ in range(10)]
    recv_models = [[] for _ in range(10)]
    for n in range(10):
        local_model[n] = net[n].heads.state_dict()
        nbr = contact[str(n)]  # the nodes n-th node contacted
        recv_models[n] = []
        for k in nbr:
            recv_models[n].append(copy.deepcopy(net[k].heads.state_dict())) # 参照渡しを防ぐために変更
            # recv_models[n].append(local_model[k])

    # mixture of models
    for n in range(10):
        update_model = copy.deepcopy(recv_models[n]) # 変更点3
        # update_model = recv_models[n]
        n_nbr = len(update_model)  # how many nodes n-th node contacted

        # put difference of n-th node models and k-th conducted node to n-th into update_model[k]
        for k in range(n_nbr):
            for key in update_model[k]:
                update_model[k][key] = recv_models[n][k][key] - local_model[n][key]

        # mix to local model
        for k in range(n_nbr):
            for key in update_model[k]:
                local_model[n][key] += (
                    update_model[k][key] * fl_coefficiency / float(n_nbr + 1)
                )
    # update nets
    for n in range(10):
        nbr = contact[str(n)]
        if len(nbr) > 0:
            net[n].heads.load_state_dict(local_model[n])


def update_nets_mobile(net, contact, fl_coefficiency):  # sはn番目の要素に複数のmodelがあるかで決まる
    local_model = [{} for _ in range(10)]
    recv_models = [[] for _ in range(10)]
    for n in range(10):
        local_model[n] = net[n].classifier[1].state_dict()
        nbr = contact[str(n)]  # the nodes n-th node contacted
        recv_models[n] = []
        for k in nbr:
            # recv_models[n].append(net[k].classifier[1].state_dict())
            recv_models[n].append(copy.deepcopy(net[k].classifier[1].state_dict())) # 参照渡しを防ぐために変更
            # recv_models[n].append(net[k].classifier[1].state_dict())

    # mixture of models
    for n in range(10):
        # update_model = copy.deepcopy(recv_models[n]) # 参照渡しを防ぐために変更
        update_model = recv_models[n]
        n_nbr = len(update_model)  # how many nodes n-th node contacted

        # put difference of n-th node models and k-th conducted node to n-th into update_model[k]
        for k in range(n_nbr):
            for key in update_model[k]:
                update_model[k][key] = recv_models[n][k][key] - local_model[n][key]

        # mix to local model
        for k in range(n_nbr):
            for key in update_model[k]:
                local_model[n][key] += (
                    update_model[k][key] * fl_coefficiency / float(n_nbr + 1)
                )
    # update nets
    for n in range(10):
        nbr = contact[str(n)]
        if len(nbr) > 0:
            net[n].classifier[1].load_state_dict(local_model[n])


def model_exchange(nets, model_name, contact, fl_coefficiency):
    if model_name == "vgg19_bn":
        update_nets_vgg(nets, contact, fl_coefficiency)
    elif model_name == "resnet_152":
        update_nets_res(nets, contact, fl_coefficiency)
    elif model_name == "vit_b16":
        update_nets_vit(nets, contact, fl_coefficiency)
    elif model_name == "mobilenet_v2":
        update_nets_mobile(nets, contact, fl_coefficiency)
