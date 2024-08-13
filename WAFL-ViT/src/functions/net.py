import torch.nn as nn
import torchvision.models as models

# prepare the model
def Net_vgg19_bn(n_output, n_middle):
    net = models.vgg19_bn(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    in_features = net.classifier[6].in_features
    net.classifier[6] = nn.Sequential(
        nn.Linear(in_features, n_middle),
        nn.ReLU(inplace=True),
        nn.Linear(n_middle, n_output),
    )
    net.avgpool = nn.Identity()
    return net


def Net_res152(n_output, n_middle):
    net = models.resnet152(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(in_features, n_middle),
        nn.ReLU(inplace=True),
        nn.Linear(n_middle, n_output),
    )
    return net


def Net_vit_b16(n_output, n_middle):
    net = models.vit_b_16(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    in_features = net.heads[0].in_features
    net.heads = nn.Sequential(
        nn.Linear(in_features, n_middle),
        nn.ReLU(inplace=True),
        nn.Linear(n_middle, n_output),
    )
    return net


def Net_mobile(n_output, n_middle):
    net = models.mobilenet_v2(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Sequential(
        nn.Linear(in_features, n_middle),
        nn.ReLU(inplace=True),
        nn.Linear(n_middle, n_output),
    )
    return net


def select_net(model_name, n_output, n_middle):
    if model_name == "vgg19_bn":
        net = Net_vgg19_bn(n_output, n_middle)
    elif model_name == "resnet_152":
        net = Net_res152(n_output, n_middle)
    elif model_name == "mobilenet_v2":
        net = Net_mobile(n_output, n_middle)
    elif model_name == "vit_b16":
        net = Net_vit_b16(n_output, n_middle)
    return net
