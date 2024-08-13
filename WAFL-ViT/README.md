# WAFL-ViT

Wireless Ad Hoc Federated Learning with Vision Transformer (WAFL-ViT). This project provides the code for the paper "Tuning Vision Transformer with Device-to-Device Communication for Targeted Image Recognition" \[1\] awarded the best paper (2nd place) at IEEE World Forum on the Internet of Things 2023.

## Architecture

<img src="./assets/architecture.png" width="75%" />

The figure shows the overview of WAFL-Vision Transformer. It is composed of multiple devices. Each of them collects data from the target environment and fine-tunes the pre-trained Vision Transformer (ViT). Here, the MLP head of ViT is replaced with another feedforward layer to fit the target task.

In our scenario, each device has ad hoc wireless interfaces and exchanges the MLP head with the neighbors through device-to-device communications (although the project code is for simulation only). This communication can be a combination of Bluetooth and Wi-Fi: i.e., finding friend workers by Bluetooth and exchanging model parameters by Wi-Fi ad hoc mode. The exchanged MLP heads are aggregated by the WAFL algorithm as below.

![Model Aggregation](./assets/model_aggregation.png)

Here, $n$ and $k$ are the devices that participated in the training. $nbr(n)$ is the set of neighbor nodes of device $n$. $W^n$ indicates the parameters of MLP head of device $n$. $\lambda$ is the coefficient which should be between 0 and 1.

## UTokyo Building Recognition Dataset

<img src="./assets/target_buildings.png" width="75%" />

As a mission-oriented task, we have generated UTokyo Building Recognition Dataset (UTBR) to provide a smart-campus service. The photos were captured by five persons with their own smartphone cameras individually. We have chosen ten buildings as the photo target, and each of the photos is labeled manually.

<img src="./assets/dataset_examples.png" width="75%" />

This figure shows the examples -- target buildings were taken from the front, back, and sides, sometimes closely, looking up, or from afar, or with a telescopic mode. Some photos contain trees, clouds, and the sun. This characteristic is not available in MNIST or CIFAR-10 datasets.

We then pre-processed the photos to distribute to virtual ten devices for both IID and Non-IID scenarios described in the previous section. The following table below shows the distributions. In the IID scenario, all the nodes have relatively the same label distributions, whereas, in the Non-IID scenario, the label distributions are dependent on the device. For example, device 0 has a larger amount of label 0 photos. Please note that even if the label distribution is IID, the devices' local photos do not cover all the scenes equally with other devices because the number of stored photos for one building is around 10 to 20.

**Table 1: Label Distribution of IID Data**
| Device | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | SUM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| 0 | 11 | 16 | 10 | 10 | 12 | 13 | 12 | 17 | 14 | 20 | 135 |
| 1 | 11 | 16 | 10 | 10 | 12 | 13 | 12 | 17 | 14 | 20 | 135 |
| 2 | 11 | 16 | 10 | 10 | 11 | 14 | 12 | 17 | 14 | 20 | 135 |
| 3 | 11 | 16 | 10 | 9 | 12 | 14 | 12 | 17 | 14 | 20 | 135 |
| 4 | 11 | 16 | 10 | 9 | 12 | 14 | 12 | 17 | 13 | 21 | 135 |
| 5 | 11 | 15 | 11 | 9 | 12 | 13 | 13 | 16 | 14 | 21 | 135 |
| 6 | 10 | 16 | 10 | 10 | 12 | 13 | 12 | 17 | 14 | 21 | 135 |
| 7 | 10 | 16 | 10 | 10 | 12 | 13 | 12 | 17 | 14 | 21 | 135 |
| 8 | 10 | 16 | 10 | 10 | 12 | 13 | 12 | 17 | 14 | 21 | 135 |
| 9 | 10 | 16 | 10 | 10 | 12 | 13 | 12 | 17 | 14 | 20 | 134 |

**Table2: Label Distribution of Non-IID Data**
| Device | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | SUM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| 0 | 44 | 4 | 5 | 4 | 8 | 4 | 8 | 8 | 11 | 12 | 108 |
| 1 | 10 | 95 | 7 | 6 | 5 | 6 | 8 | 9 | 6 | 15 | 167 |
| 2 | 4 | 7 | 59 | 3 | 8 | 10 | 5 | 9 | 11 | 11 | 127 |
| 3 | 5 | 7 | 4 | 45 | 8 | 5 | 3 | 9 | 14 | 10 | 110 |
| 4 | 8 | 8 | 3 | 8 | 52 | 9 | 9 | 14 | 5 | 10 | 126 |
| 5 | 4 | 7 | 2 | 6 | 8 | 76 | 9 | 11 | 8 | 9 | 140 |
| 6 | 7 | 6 | 7 | 7 | 4 | 4 | 56 | 7 | 9 | 13 | 120 |
| 7 | 6 | 10 | 2 | 6 | 9 | 4 | 7 | 84 | 6 | 11 | 145 |
| 8 | 5 | 5 | 7 | 6 | 6 | 6 | 7 | 8 | 59 | 16 | 125 |
| 9 | 13 | 10 | 5 | 6 | 11 | 9 | 9 | 10 | 10 | 98 | 181 |

You can access our dataset from [this link](https://drive.google.com/file/d/1GKbMyfAkvCVT1a6g2KyvkC3MYxf5VPrZ/view).

## Concept of this folder

- In this project, we aimed to train models that is larger and can solve more complicated tasks than that of the project in WAFL-MLP folder.

- We combined transfer learning and WAFL to allow real-domain complex images to input the model with reasonable computational cost and network load.

- We compared multiple models and clarified VisionTransformer experimentally showed the best result.

## Structure of this directory

```plain text
|- WAFL-ViT
|   |-data
|   |   |-val
|   |   |   |-0
|   |   |   |-1
|   |   |   
|   |   |-train
|   |   |   |-0
|   |   |   |-1
|   |   |   
|   |   |-non-IID_filter
|   |   |   |-mean.pth (Mean of images each node has)
|   |   |   |-std.pth (Standard diviation of images each node has)
|   |   |
|   |   |-contact_pattern
|   |   |   |-pattern file (Describe how to nodes contact each other)
|   |   |   
|   |   |-test_mean_and_std
|   |   |   |-mean.pth (Mean of images use for validation)
|   |   |   |-std.pth (Standard diviation of images use for validation)
|   |
|   |-src
|   |   |-functions
|   |   |   |-definitions 
|   |   |   |
|   |   |-main.py
|   |   
|   |-results
|   |   |-20240515 (Results. You can adjust its name in the program.)
|   |   |   |-log.txt
|   |   |   |-params
|   |   |   |   |-model_parameters
|   |   |   |   |-histories (Trend data in the training)
|   |   |   |-images
|   |   |   |   |-latent_space (Latent space of the model at epoch [number] for node [node_id])
|   |   |   |   |   |-ls-epoch{number}-node{node_id}.png
|   |   |   |   |-normalized_confusion_matrix (Confusion matrix of the model at epoch [number] for node [node_id])
|   |   |   |   |   |-normalized-cm-epoch{number}-node{node_id}.png
|   |   |   |   |-acc.png (Trend in accuracy)
|   |   |   |   |-loss.png (Trend in loss)
```

## Data installation

<!--
![System overview](./assets/dataset_abstract.png)

In this project, we created and utilized the dataset which consist of  images of several buildings at the University of Tokyo.
The mapping between labels and buildings is shown in the image above.
-->

You can access our dataset from [this link](https://drive.google.com/file/d/1GKbMyfAkvCVT1a6g2KyvkC3MYxf5VPrZ/view).

After downloading zip file, please extract its contents into the `WAFL-ViT/data` directory of the project root.
If you're using the command line and are in the project root (`wafl` directory), you can use the following command to extract the files:

``` Linux
cd WAFL-ViT
mv [downloaded file path] ./
unzip -q vit_data.zip
mv vit_data/* data/
rm -r vit_data*
```

Regarding usage or licensing of this dataset, please refer to the `LICENSE` in the project root.

## Usage

### Module installation

This code has been tested and verified to work with Python 3.11.4 and CUDA 11.4.
The specific versions of key dependencies used in our test environment are listed in the `requirements.txt` file.

However, please note that you may need to adjust the versions, especially for `torch` and `torchvision`, to match your specific environment and CUDA version.

After ensuring versions of required dependencies, install them by following commands:

```Linux
pip install -r requirements.txt
```

If you encounter any issues, you may need to modify the versions in `requirements.txt` to suit your specific setup. In particular, ensure that the `torch` and `torchvision` versions are compatible with your CUDA installation if you're using GPU acceleration.

### How to run

To start the training and store its results, please follow these steps:

1. Ensure the dataset is correctly located in the expected directory.

    ```plain text
    |- WAFL-ViT
    |   |- data
    |   |   |-val
    |   |   |   |-0
    |   |   |   |-1
    |   |   |   
    |   |   |-train
    |   |   |   |-0
    |   |   |   |-1
    |   |   |   
    |   |   |-non-IID_filter
    |   |   |
    ```

2. Check that all required dependencies are correctly installed.
3. Move to the `src` directory:
  
    ```Linux
    cd src
    ```

4. Prepare contact patterns and filters:

    ```Linux
    python utils/generate_contact_pattern.py
    python utils/generate_nonIID_filter.py
    ```

5. Review and adjust the experimental settings in the config file(`src/config.json`). For detailed instructions on how to write and configure the setting file, please refer to the [Configuration File Guide](#configuration-file-guide):

    ```Linux
    vim config.json  # or use any text editor of your choice
    ```

6. Start the training process:

    ```Linux
    python main.py
    ```

7. Verify the start of the training process:

   After starting the training process, you can find that log in `results/{result folder name}/log.txt`.

8. Confirm the log file(`results/{result folder name}/log.txt`):

   You can find your experimental conditions in the log file.

## Output

### Final model accuracy and loss

You can check the final model loss and accuracy of all nodes as a result of the model training.
These scores are recorded in the log file(`results/{result folder name}/log.txt`) as shown in the following example:

```plain text
Initial Epoch (node0): Loss: 3.75950 Accuracy: 0.44291
Final Epoch (node0): Loss: 0.47345 Accuracy: 0.86851
Initial Epoch (node1): Loss: 3.84433 Accuracy: 0.41522
Final Epoch (node1): Loss: 0.47143 Accuracy: 0.86851
Initial Epoch (node2): Loss: 4.53923 Accuracy: 0.41522
Final Epoch (node2): Loss: 0.47127 Accuracy: 0.86851
...
```

Additionally, you can confirm the average accuracy and its standard deviation across all nodes for the last 10 epochs.
These statistics are also available in the same file, presented as follows:

```plain text
the average of the last 10 epoch: 0.8694059976931949
the std of the last 10 epoch: 0.004650926644612355
```

### Trend graph

After model training is successfully completed, trend graphs for all nodes are created in `results/{result folder name}/images/acc.png (or loss.png)`.

### Images of confusion matrix and latent space

Once 75% of training process is complete, images of the confusion matrix and latent space of models are generated every 50 epochs.
These images are stored in the following directories:

- Confusion matrices: `results/{result folder name}/images/normalized_confusion_matrix`
- Latent space visualizations: `results/{result folder name}/images/latent_space`

## Configuration File Guide

You can configure the parameters and settings for the experiment with `src/config.json`.
This file allows you to easily customize the training process.

Below are the fields of `config.json`.
### model

`model_name`(str): The model which you use in the experiment. This parameter should be either of [`vgg19_bn`, `mobilenet_v2`, `resnet_152`, `vit_b16`].

`n_middle`(int): The number of input units for the added classification layer. To make use of the WAFL's parameter aggregation, we added another layer for the classification layer.

### data

`n_node`(int): The number of nodes that participate in the training process of WAFL.

### gpu

`device`(str): Set the name of GPU which you want to use. (e.g. "cuda:0", "cuda:1")

`transform_on_gpu`(boolean): This option speed up the training process by loading the images to GPU in advance & conduct data-augmentation in GPU. 
Set this option to `true` to enable the feature. 
Note that this will consume more GPU memory. 
For detailed explanation, please refer to the `src/functions/mydataset.py`.

### mode

`self_train_only`(boolean): We support the training mode which only conduct self-training phase in WAFL. 
Set this option to `true` only when you want to try self-training phase but do not want to proceed to the subsequent collaborative training phase.

### self_training

This section configure the settings in the self-training phase.

`epochs`(int): Maximum epoch

`learning_rate`(float): Learning rate 

`optimizer_name`(str): Choose optimizer from [`SGD`, `Adam`].

`momentum`(float): Momentum of the optimizer.

`use_scheduler`(boolean): If you want to use schedulers in the self-training phase, set this option `true`. 
We used `StepLR` which means step decay of learning rate.

`scheduler_rate`(float): This option specifies the multiplicative factor by which the learning rate is reduced.

`scheduler_step`(int): This option specifies the number of epochs after which the learning rate is decreased.

### collaborative_training

`fl_coefficient`(float): Aggregation coefficient in wafL.

Please refer to the [self-training](#self_training) section for the other configurations.

### non_IID_filter

You can use non-IID filters to simulate the non-IID scenarios. Set `use_noniid_filter` to `true` to use non-IID filter.

### contact_pattern

You have to prepare moving pattern of nodes that participate in the collaborative training. See `src/utils/generate_contact_pattern.py` & `visualize_contact_pattern.py` for more details.

## References

\[1\] Hideya Ochiai, Atsuya Muramatsu, Yudai Ueda, Ryuhei Yamaguchi, Kazuhiro Katoh, and Hiroshi Esaki, "[Tuning Vision Transformer with Device-to-Device Communication for Targeted Image Recognition](https://ieeexplore.ieee.org/abstract/document/10539480)", IEEE World Forum on Internet of Things, 2023. 
