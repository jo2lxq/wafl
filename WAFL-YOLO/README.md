# WAFL-YOLO

Wireless Ad Hoc Federated Learning with YOLOv9 (WAFL-YOLO). These codes are edited versions of [the original YOLOv9 codes](https://github.com/WongKinYiu/yolov9) to make them compatible with WAFL.

## Model exchange and aggregation

In this study, we verified three learning and model exchange methods. The first is Full Parameter Exchange (FPE), in which all parameters are learned and exchanged. The second is Detection Head Exchange (DHE), in which only the paramterers in Detection Head are learned and exchanged. The third is Head Last Exchange (HLE), in which only the last convolution layers in Detection Head are learned and exchanged.

Also, in this scenario, all the devices are assumed to be fixed and exchange their model with neighboring devices based on topology. We verified three types of topology : line, tree, and ringstar.

## Data preparation

We created the target dataset by selecting 10 categories from the Open Image Dataset and used it. Please download the custom dataset from [here](https://drive.google.com/file/d/1dFmatagowqRz7zAf0sZNxREe0sp5kX4Z/view?usp=sharing) and set it in the `data` directory. We expect the directory strcture to be the following.
```
WAFL-YOLO/data/custom_yolo/
  train # train images and labels
  val # val images and labels
```

## Usage

### Module installation

This code has been tested and verified to work with Python 3.12.3 and CUDA 12.4. The specific versions of key dependencies used in our test environment are listed in the `requirements.txt` file.

However, please note that you may need to adjust the versions, especially for `torch` and `torchvision`, to match your specific environment and CUDA version.

After ensuring versions of required dependencies, install them by following commands:

```
pip install -r requirements.txt
```

If you encounter any issues, you may need to modify the versions in `requirements.txt` to suit your specific setup. In particular, ensure that the `torch` and `torchvision` versions are compatible with your CUDA installation if you're using GPU acceleration.

### Change directory
We expect you to execute codes in the `src` directory, so please move there like:

```
cd src
```

### Pretrained-model preparation
Please download pretrained YOLOv9-c model from [here](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt) and set it in `src` directory.

For more information about the model, please refer to [here](https://github.com/WongKinYiu/yolov9).

### Training

To train the models, please exectute `train_wafl.py` like:

```
python train_wafl.py \
    --batch 2 \
    --preself-epochs 30 \
    --epochs 90 \
    --weights "yolov9-c.pt" \
    --num_clients 10 \
    --noniid_ratio 90 \
    --topology "line"
```

After training, `outputs` directory is created and it includes `results.csv` and checkpoints. The following explains some config parameters.

#### weights
The path to the file of model parameters. The parameters set here will be used as the initial values when training starts.

#### num_clients
The number of clients which participates in the collaborative learning.

#### noniid_ratio
The percentage of each client's class in the noniid scenario. 

#### topology
Select from `"line"`, `"tree"` and `"ringstar"`. `"line"` is set as the default.

#### iid_setting
Set with boolean `True` and training starts with iid scenario. `False` is set as the default. 

#### dhe
Set with boolean `True` and training starts with method DHE. `False` is set as the default. Training starts with method FPE in that case.

#### hle
Set with boolean `True` and training starts with method HLE. `False` is set as the default. Training starts with method FPE in that case.

#### detail_log
Displays the training log in more detail if `True` set. `False` is set as the default.

### Visualization

After training, `outputs` directory is made. (If the `outputs` directory already exists, a new directory named `outputs1` will be created. If `outputs1` exists, `outputs2` will be created, and so on). It includes the results of training. If you want to visualize the trends in mAP, you can run `mAP_plot.py` like: 

```
python ./utils/bin/mAP_plot.py --dirname "outputs"
```
Please specify the directory name for reading data and saving plot results with `--dirname`.

If you want to check the image with infered bounding box, you can use `box_visualization.py` like:

```
python ./utils/bin/box_visualization.py \
    --source '../data/custom_yolo/val/images/0957f84aecdf874d.jpg' \
    --weights 'outputs/weights/node1/last.pt' \
    --conf-thres 0.3 \
    --project "outputs"
```
You can select the image with `--source`, the model with `--weights`, the confidence score threshold with `--conf-thres`, and in which directory to save with `--project`. Please specify the filename in `data/custom_yolo/val/images` for `--source`.
