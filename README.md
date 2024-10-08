# Welcome to Wireless Ad Hoc Federated Learning (WAFL)
This space provides the code for Wireless Ad Hoc Federated Learning (WAFL) -- A Fully Autonomous Collaborative Learning with Device-to-Device Communication.

As of Aug 2024, this repository contains the following two projects.
1. [WAFL-MLP](./WAFL-MLP/): The most basic codes with a fully-connected neural network for starters. You can learn what the WAFL is.
2. [WAFL-ViT](./WAFL-ViT/): WAFL with Vision Transformer for image recognition.

## What is WAFL?

Wireless ad hoc federated learning (WAFL) allows collaborative learning via device-to-device communications organized by the devices physically nearby. Here, each device has a wireless interface and can communicate with each other when they are within the radio range. The devices are expected to move with people, vehicles, or robots, producing opportunistic contacts with each other.

<img src="./WAFL-MLP/assets/wafl_contact_model_aggregation.png">

Each device trains a model individually with the local data it has. When a device encounters another device, they exchange their local models with each other through the ad hoc communication channel. Then, the device aggregates the models into a new model, which is expected to be more general compared to the locally trained models. With an adjustment process of the new model with the local training data, they repeat this process during they are in contact. Please note that there is no third-party server operated for the federation among multi-vendor devices.

**(*) Click [here](./assets/WAFL_project_en.pdf) to see the white paper.** <br /><br />
[<img src="./assets/WAFL_project.png">](./assets/WAFL_project_en.pdf)

## WAFL-Variations

| WAFL-Vision Transformer | WAFL-Autoencoder | Multi-Task WAFL |
| --- | --- | --- |
| [<img src="./assets/WAFL_ViT.png">](./assets/WAFL_ViT.pdf)　|　[<img src="./assets/WAFL_Autoencoder.png">](./assets/WAFL_Autoencoder.pdf) | [<img src="./assets/MT_WAFL.png">](./assets/MT_WAFL.pdf) |
|This work studies the use of the Vision Transformer for image recoginition tasks. In this scenario, each device has a whole ViT model and exchanges a part of the model parameters with neighbors through device-to-device communications during its collaborative training phase. This work demonstrates that WAFL-ViT works efficiently for developing image recognition models.| This work studies the scenario that multiple IoT devices are installed in a remote site, such as in a building, and that they collaboratively detect anomalies from the observations. They are expected (1) to learn the normal features from daily observations and (2) to detect anomalies when they occurred. | People make various reactions to a presented object unconsciously. When looking at an object, some people may say about the object itself (e.g., Apple), others may say about the color (e.g., Purple), and others may say about the size (e.g., Big). By integrating these various reactions from many user sources, a machine will be able to provide answers at the same time, working as a brainstormer for a presented object. MT-WAFL allows to train such models collaboratively.|

## Publications
\[1\] **Hideya Ochiai, Yuwei Sun, Qingzhe Jin, Nattanon Wongwiwatchai, Hiroshi Esaki, "Wireless Ad Hoc Federated Learning: A Fully Distributed Cooperative Machine Learning" in May 2022 (https://arxiv.org/abs/2205.11779).** 

\[2\] Naoya Tezuka, Hideya Ochiai, Yuwei Sun, Hiroshi Esaki, "Resilience of Wireless Ad Hoc Federated Learning against Model Poisoning Attacks", IEEE International Conference on Trust, Privacy and Security in Intelligent Systems, and Applications (TPS-ISA), 2022 (https://ieeexplore.ieee.org/abstract/document/10063735).

\[3\] Eisuke Tomiyama, Hiroshi Esaki, Hideya Ochiai, "WAFL-GAN: Wireless Ad Hoc Federated Learning for Distributed Generative Adversarial Networks", IEEE International Conference on Knowledge and Smart Technology, 2023 (https://ieeexplore.ieee.org/document/10086811).

\[4\] Hideya Ochiai, Riku Nishihata, Eisuke Tomiyama, Yuwei Sun, and Hiroshi Esaki, "Detection of Global Anomalies on Distributed IoT Edges with Device-to-Device Communication", ACM MobiHoc, 2023 (https://dl.acm.org/doi/abs/10.1145/3565287.3616528).

\[5\] **Hideya Ochiai, Atsuya Muramatsu, Yudai Ueda, Ryuhei Yamaguchi, Kazuhiro Katoh, and Hiroshi Esaki, "Tuning Vision Transformer with Device-to-Device Communication for Targeted Image Recognition", IEEE World Forum on Internet of Things (WF-IoT), 2023 (Best Paper Award) (https://ieeexplore.ieee.org/document/10539480).**

\[6\] Ryusei Higuchi, Hiroshi Esaki, and Hideya Ochiai, "Personalized Wireless Ad Hoc Federated Learning for Label Preference Skew", IEEE World Forum on Internet of Things (WF-IoT), 2023 (https://ieeexplore.ieee.org/document/10539563).

\[7\] Yusuke Sugizaki, Hideya Ochiai, Muhammad Asad, Manabu Tsukada, and Hiroshi Esaki, "Wireless Ad-Hoc Federated Learning for Cooperative Map Creation and Localization Models", IEEE World Forum on Internet of Things (WF-IoT), 2023 (https://ieeexplore.ieee.org/document/10539517).

\[8\] Koshi Eguchi, Hideya Ochiai, and Hiroshi Esaki, "MemWAFL: Efficient Model Aggregation for Wireless Ad Hoc Federated Learning in Sparse Dynamic Networks", IEEE Future Networks World Forum, 2023 (https://ieeexplore.ieee.org/document/10520500).

\[9\] Yoshihiko Ito, Hideya Ochiai, and Hiroshi Esaki, "Self-Organizing Hierarchical Topology in Peer-to-Peer Federated Learning: Strategies for Scalability, Robustness, and Non-IID Data", IEEE Future Networks World Forum, 2023 (https://ieeexplore.ieee.org/document/10520530).

\[10\] Ryusei Higuchi, Hiroshi Esaki, and Hideya Ochiai, "Collaborative Multi-Task Learning across Internet Edges with Device-to-Device Communications", IEEE Cybermatics Congress (SmartData), 2023 (https://ieeexplore.ieee.org/document/10501784).

\[11\] Yudai Ueda, Hideya Ochiai, Hiroshi Esaki, "Device-to-Device Collaborative Learning for Self-Localization with Previous Model Utilization", IEEE International Conference on Knowledge and Smart Technology, 2024 (https://ieeexplore.ieee.org/document/10499694).

\[12\] Atsuya Muramatsu, Hideya Ochiai, Hiroshi Esaki, "Tuning Personalized Models by Two-Phase Parameter Decoupling with Device-to-Device Communication", IEEE International Conference on Knowledge and Smart Technology, 2024 (Best Paper Award) (https://ieeexplore.ieee.org/document/10499649).

\[13\] Ryuhei Yamaguchi, Hideya Ochiai, "Tuning Detection Transformer with Device-to-Device Communication for Mission-Oriented Object Detection", IEEE WiMob CWN workshop, 2024 (in press).

\[14\] Ryusei Higuchi, Hiroshi Esaki, Hideya Ochiai, "Neuron Personalization of Collaborative Federated Learning via Device-to-Device Communications", IEEE WiMob CWN workshop, 2024 (in press).
