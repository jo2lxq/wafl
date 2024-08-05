# WAFL-MLP

Wireless Ad Hoc Federated Learning (WAFL). This project provides the code for "Wireless Ad Hoc Federated Learning: A Fully Distributed Cooperative Machine Learning". MLP stands for multi-layer perceptron -- one of the most basic neural networks for beginners.

## Introduction
Privacy-sensitive data is stored in autonomous vehicles, smart devices, or sensor nodes that can move around and can make opportunistic contact with each other. Geographical locations, private photos, healthcare signals, and power consumption of homes are examples. Federation among such nodes was mainly discussed in the context of federated learning (FL) in many works with a centralized parameter server. However, because of multi-vendor issues, they do not want to rely on a specific server operated by a third party for this purpose but want to directly interact with each other in an ad hoc manner only when they have in contact just as cooperative intelligent transport systems do.

Wireless ad hoc federated learning (WAFL) allows collaborative learning with device-to-device communications organized by the devices physically nearby. Here, each node has a wireless interface and can communicate with each other when they are within the radio range. The nodes are expected to move with people, vehicles, or robots, producing opportunistic contacts with each other.

## Architecture
<img src="./assets/wafl_overview.png" width="75%">

We start the discussion from the most basic peer-to-peer case as shown in the figure. In this scenario, each node trains a model individually with the local data it has. Here, node 1 can be a Toyota car, and node 2 can be a Mitsubishi car. When a node encounters another node, they exchange their local models with each other through the ad hoc communication channel. Then, the node aggregates the models into a new model, which is expected to be more general compared to the locally trained models. With an adjustment process of the new model with the local training data, they repeat this process during they are in contact. Please note that there is no third-party server operated for the federation among multi-vendor devices.

As WAFL does not collect data from users, the distributions of the data on individual nodes are not the same; e.g., a user has a larger portion of photos of hamburgers, but another user has a larger portion of dumplings based on their preferences or circumstances. This is a well-known problem of conventional federated learning as ``user data is not independent and identically distributed (Non-IID)''. The challenge is to develop a general model which does not over-fit into specific user data on the fully distributed, or partially-connected environment. 

## Model Aggregation via Ad Hoc Contacts

<img src="./assets/wafl_contact_model_aggregation.png">

Model exchange and aggregation with encountered smart devices in wireless ad hoc federated learning (WAFL). The nodes exchange and aggregate their models among the nodes encountered in an ad hoc manner. The initial models are trained too specific to their local Non-IID data, but in the long run, many contacts allow the mixture of locally trained models, making them more generalized.

## How to Run

TODO. Descriptions of How to run will be given.

## RWP0500

<img src="./assets/rwp0500.gif">

TODO. Descriptions of RWP0500 will be given.

## Contact Pattern for Simulation
<img src="./assets/contact_pattern.png">

## Confusion Matrix

<img src="./assets/confusion_matrix.png">


## References 
\[1\] Hideya Ochiai, Yuwei Sun, Qingzhe Jin, Nattanon Wongwiwatchai, Hiroshi Esaki, "Wireless Ad Hoc Federated Learning: A Fully Distributed Cooperative Machine Learning" in May 2022 (https://arxiv.org/abs/2205.11779). 
