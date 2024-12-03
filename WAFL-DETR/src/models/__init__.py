# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build, build_wafl


def build_model(args):
    return build(args)

def build_wafl_models(args):
    return build_wafl(args)