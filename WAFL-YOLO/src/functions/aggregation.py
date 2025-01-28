import json
import torch

def model_aggregate(models, topology="line", private_param=[], coef=1.0):
    with open('/home/u01071/detr/graph.json') as f:
        g_dict = json.load(f)
    graph = g_dict[topology]
    model_dicts = []

    for i in range(len(models)):
        model_dicts.append(models[i].state_dict())
    param_diff_dicts = [{k: torch.zeros_like(v) for k, v in model_dicts[i].items()} for i in range(len(models))]
    updated_model_dicts = [{} for _ in range(len(models))]

    for i in range(len(models)):
        for nbr in graph[i]:
            for param_key in model_dicts[i]:
                param_diff_dicts[i][param_key] += (model_dicts[nbr][param_key] - model_dicts[i][param_key])
        for param_key in model_dicts[i]:
            if param_key in private_param:
                updated_model_dicts[i][param_key] = model_dicts[i][param_key]
            else:
                updated_model_dicts[i][param_key] = model_dicts[i][param_key] + coef * param_diff_dicts[i][param_key] / (len(graph[i]) + 1)
    
    for i in range(len(models)):
        models[i].load_state_dict(updated_model_dicts[i])
    
    return models

def model_aggregate_fl(models, private_param=[]):
    model_dicts = []
    for i in range(len(models)):
        model_dicts.append(models[i].state_dict())
    global_model_dict = {k: torch.zeros_like(v) for k, v in model_dicts[0].items()}
    for param_key in global_model_dict:
        if param_key in private_param:
            global_model_dict[param_key] = model_dicts[0][param_key]
        else:
            for i in range(len(models)):
                global_model_dict[param_key] += (model_dicts[i][param_key] / len(models))

    for i in range(len(models)):
        models[i].load_state_dict(global_model_dict)

    return models
