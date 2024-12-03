#!/bin/bash
#SBATCH -p v
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem 16GB

source $HOME/anaconda3/bin/activate

python main_wafl.py \
    --dataset_file "custom" \
    --coco_path "../../data/custom/" \
    --output_dir "outputs" \
    --resume "detr-r50_no-class-head.pth" \
    --num_classes 10 \
    --preself_epochs 100 \
    --epochs 200 \
    --batch_size 2 \
    --num_clients 10 \
    --noniid_ratio 90 \
    --lr 1e-5 \
    --lr_backbone 1e-6