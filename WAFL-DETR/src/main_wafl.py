# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset

from functions.engine_wafl import evaluate, train_one_epoch
from functions.aggregation import model_aggregate

from models import build_wafl_models

from util import generate_noniid, generate_iid


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=1000, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--tle', type=bool, default=None, help="frozen backbone parameters")
    parser.add_argument('--he', type=bool, default=None, help="frozen all parameters except head")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--num_classes', default=None, type=int,
                        help='#classes in your dataset, which can override the value hard-coded in file models/detr.py')
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # WAFL training parameters
    parser.add_argument('--num_clients', default=1, type=int, help='number of clients participating in wafl')
    parser.add_argument('--noniid_ratio', default=90, type=int)
    parser.add_argument('--iid_setting', default=False, type=bool)
    parser.add_argument('--no_exchange', default=False, type=bool)
    parser.add_argument('--preself_epochs', default=50, type=int)
    parser.add_argument('--topology', default="line", type=str)
    parser.add_argument('--resume_from_preselftrained', type=str, default='')
    parser.add_argument('--resume_from_wafl', type=str, default='')

    return parser


def main(args):
    args.distributed = None

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    models, criterion, postprocessors = build_wafl_models(args)

    for i in range(args.num_clients):
        models[i] = models[i].to(device)
    
    private_param = []
    if args.tle:
        print("TLE mode")
        for i in range(len(models)):
            for name, p in models[i].named_parameters():
                if 'backbone' in name:
                    p.requires_grad=False
                    private_param.append(name)

    if args.he:
        print("HE mode")
        for i in range(len(models)):
            for name, p in models[i].named_parameters():
                if ('class_embed' in name) or ('bbox_embed' in name) or ('query_embed' in name):
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    n_parameters = [sum(p.numel() for p in models[i].parameters() if p.requires_grad) for _ in range(args.num_clients)]
    print('number of params:', n_parameters)

    all_param_dicts = []
    for i in range(args.num_clients):
        param_dicts =  [
            {"params": [p for n, p in models[i].named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in models[i].named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        all_param_dicts.append(param_dicts)
    
    optimizers = []
    lr_schedulers = []
    for i in range(args.num_clients):
        optimizer = torch.optim.AdamW(all_param_dicts[i], lr=args.lr,
                                  weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        optimizers.append(optimizer)
        lr_schedulers.append(lr_scheduler)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.iid_setting:
        filter_path = f'../config/filter_s{args.seed}.pt'
        if not os.path.isfile(filter_path):
            generate_iid.generate(dataset_train, args.batch_size, args.num_clients, args.num_classes, filter_path, args.seed)
    else:
        filter_path = f'../config/filter_r{args.noniid_ratio}_s{args.seed}.pt'
        if not os.path.isfile(filter_path):
            generate_noniid.generate(dataset_train, args.batch_size, args.num_clients, args.num_classes, filter_path, args.noniid_ratio, args.seed)

    indices = torch.load(filter_path)
    subsets = [Subset(dataset_train, indices[i]) for i in range(args.num_clients)]
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loaders_train = [DataLoader(subsets[i], batch_size=args.batch_size, collate_fn=utils.collate_fn, num_workers=args.num_workers) \
                          for i in range(args.num_clients)]
    
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.output_dir:
        os.makedirs(f'{args.output_dir}', exist_ok=True)
        for i in range(args.num_clients):
            os.makedirs(f'{args.output_dir}/node{i}', exist_ok=True)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        
        for i in range(args.num_clients):
            models[i].load_state_dict(checkpoint['model'], strict=False)
    
    if args.resume_from_preselftrained or args.start_epoch != 0:
        for i in range(args.num_clients):
            if args.resume_from_preselftrained:
                resume_path = Path(args.resume_from_preselftrained)
                checkpoint = torch.load(resume_path / 'outputs' / f'node{i}' / f'checkpoint{str(args.preself_epochs - 1).zfill(4)}_preself_train.pth', map_location='cpu')
                models[i].load_state_dict(checkpoint['model'], strict=False)
            else:
                resume_path = Path(args.resume_from_wafl)
                checkpoint = torch.load(resume_path / 'outputs' / f'node{i}' / f'checkpoint{str(args.start_epoch - 1).zfill(4)}.pth', map_location='cpu')
                models[i].load_state_dict(checkpoint['model'], strict=False)
                optimizers[i].load_state_dict(checkpoint['optimizer'])
                lr_schedulers[i].load_state_dict(checkpoint['lr_scheduler'])

    if (not args.resume_from_preselftrained) and (args.start_epoch == 0):
        print("Start pre-self training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.preself_epochs):
            train_stats = train_one_epoch(
                models, criterion, data_loaders_train, optimizers, device, epoch,
                args.clip_max_norm)
            if args.output_dir:
                for i in range(args.num_clients):
                    checkpoint_paths = [output_dir / f'node{i}' / 'checkpoint_preself_train.pth']
                    # extra checkpoint before LR drop and every 10 epochs
                    if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                        checkpoint_paths.append(output_dir / f'node{i}' / f'checkpoint{epoch:04}_preself_train.pth')
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': models[i].state_dict(),
                            'optimizer': optimizers[i].state_dict(),
                            'lr_scheduler': lr_schedulers[i].state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)

            test_stats, coco_evaluators = evaluate(
                models, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )

            log_stats = [{**{f'train_{k}': v for k, v in train_stats[i].items()},
                        **{f'test_{k}': v for k, v in test_stats[i].items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters[i]} for i in range(args.num_clients)]

            if args.output_dir and utils.is_main_process():
                for i in range(args.num_clients):
                    with (output_dir / f"node{i}" / "log_preself_train.txt").open("a") as f:
                        f.write(json.dumps(log_stats[i]) + "\n")

                # for evaluation logs
                if coco_evaluators is not None:
                    for i in range(args.num_clients):
                        (output_dir / f'node{i}' / 'eval').mkdir(exist_ok=True)
                        if "bbox" in coco_evaluators[i].coco_eval:
                            filenames = ['latest_preself_train.pth']
                            if epoch % 50 == 0:
                                filenames.append(f'{epoch:03}_preself_train.pth')
                            for name in filenames:
                                torch.save(coco_evaluators[i].coco_eval["bbox"].eval,
                                        output_dir / f'node{i}' / 'eval' / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time (pre-self training) {}'.format(total_time_str))
    else:
        print('pre-self training is skipped')

    print("Start WAFL training")
    start_time = time.time()
    if args.start_epoch != 0:
        print(f"resume from epoch {args.start_epoch}")
    for epoch in range(args.start_epoch, args.epochs):
        if args.no_exchange:
            pass
        else:
            models = model_aggregate(models, args.topology, private_param)
        train_stats = train_one_epoch(
            models, criterion, data_loaders_train, optimizers, device, epoch,
            args.clip_max_norm)
        for i in range(args.num_clients):
            lr_schedulers[i].step()
        if args.output_dir:
            for i in range(args.num_clients):
                checkpoint_paths = [output_dir / f'node{i}' / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 10 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                    checkpoint_paths.append(output_dir / f'node{i}' / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': models[i].state_dict(),
                        'optimizer': optimizers[i].state_dict(),
                        'lr_scheduler': lr_schedulers[i].state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

        test_stats, coco_evaluators = evaluate(
            models, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = [{**{f'train_{k}': v for k, v in train_stats[i].items()},
                     **{f'test_{k}': v for k, v in test_stats[i].items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters[i]} for i in range(args.num_clients)]

        if args.output_dir and utils.is_main_process():
            for i in range(args.num_clients):
                with (output_dir / f"node{i}" / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats[i]) + "\n")

            # for evaluation logs
            if coco_evaluators is not None:
                for i in range(args.num_clients):
                    (output_dir / f'node{i}' / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluators[i].coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluators[i].coco_eval["bbox"].eval,
                                    output_dir / f'node{i}' / 'eval' / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time (WAFL or FL training) {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)