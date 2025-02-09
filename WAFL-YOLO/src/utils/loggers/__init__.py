import os
import warnings
from pathlib import Path

import pkg_resources as pkg
import torch

from utils.general import LOGGER, colorstr, cv2
from utils.plots import plot_images, plot_labels, plot_results
from utils.torch_utils import de_parallel

LOGGERS = ('csv', 'tb', 'wandb', 'clearml', 'comet')  # *.csv, TensorBoard, Weights & Biases, ClearML
RANK = int(os.getenv('RANK', -1))


class Loggers():
    # YOLO Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.plots = not opt.noplots  # plot results
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = [
            'train/box_loss',
            'train/cls_loss',
            'train/dfl_loss',  # train loss
            'metrics/precision',
            'metrics/recall',
            'metrics/mAP_0.5',
            'metrics/mAP_0.5:0.95',  # metrics
        ]  # params
        self.best_keys = ['best/epoch', 'best/precision', 'best/recall', 'best/mAP_0.5', 'best/mAP_0.5:0.95']
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

    @property
    def remote_dataset(self):
        # Get data_dict if custom dataset artifact link is provided
        data_dict = None

        return data_dict

    def on_train_start(self):
        pass

    def on_pretrain_routine_start(self):
        pass

    def on_pretrain_routine_end(self, labels, names):
        # Callback runs on pre-train routine end
        if self.plots:
            plot_labels(labels, names, self.save_dir)
            paths = self.save_dir.glob('*labels*.jpg')  # training labels

    def on_train_batch_end(self, model, ni, imgs, targets, paths, vals):
        log_dict = dict(zip(self.keys[0:3], vals))
        # Callback runs on train batch end
        # ni: number integrated batches (since train start)
        if self.plots:
            if ni < 3:
                f = self.save_dir / f'train_batch{ni}.jpg'  # filename
                plot_images(imgs, targets, paths, f)
                if ni == 0 and self.tb and not self.opt.sync_bn:
                    log_tensorboard_graph(self.tb, model, imgsz=(self.opt.imgsz, self.opt.imgsz))
            if ni == 10:
                files = sorted(self.save_dir.glob('train*.jpg'))

    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        pass

    def on_val_start(self):
        pass

    def on_val_image_end(self, pred, predn, path, names, im):
        # Callback runs on val image end
        pass

    def on_val_batch_end(self, batch_i, im, targets, paths, shapes, out):
        pass

    def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix):
        # Callback runs on val end
        pass

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi, node_i):
        # Callback runs at the end of each fit (train+val) epoch
        x = dict(zip(self.keys, vals))
        dir = self.save_dir / f'node{node_i}'
        os.makedirs(dir, exist_ok=True)
        if self.csv:
            file = self.save_dir / f'node{node_i}' / 'results.csv'
            n = len(x) + 1  # number of cols
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback runs on model save event
        pass

    def on_train_end(self, last, best, epoch, results):
        # Callback runs on training end, i.e. saving best model
        if self.plots:
            plot_results(file=self.save_dir / 'results.csv')  # save results.png
        files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter
        self.logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")

    def on_params_update(self, params: dict):
        # Update hyperparams or configs of the experiment
        pass
