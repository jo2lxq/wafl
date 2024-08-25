import json
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from functions.model_exchange import *
from functions.mydataset import *
from functions.net import *
from functions.train_functions import *
from functions.visualize import *
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchinfo import summary
from torchvision.datasets import ImageFolder

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

if __name__ == "__main__":
    ## 1. Initial settings and paths
    # path
    main_path = os.path.dirname(
        os.path.abspath(__file__)
    )  # Absolute path to main.py. Note that "main_path" does not include file name i.e. "main.py".
    data_path = os.path.normpath(os.path.join(main_path, "../data"))
    project_path = os.path.normpath(os.path.join(main_path, "../results"))
    noniid_filter_path = os.path.normpath(
        os.path.join(main_path, "../data/non-IID_filter")
    )
    contact_pattern_path = os.path.normpath(
        os.path.join(main_path, "../data/contact_pattern")
    )
    mean_and_std_path = os.path.normpath(
        os.path.join(main_path, "../data/test_mean_and_std")
    )
    config_path = os.path.join(main_path, "config.json")
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "val")
    meant_file_path = os.path.join(
        mean_and_std_path, "test_mean.pt"
    )  ## Same for IID and NonIID
    stdt_file_path = os.path.join(
        mean_and_std_path, "test_std.pt"
    )  ## Same for IID and NonIID

    ## 2. Variable definitions (modify here)
    # 2.1 Output file name
    cur_index = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # cur_index = "Sample"
    cur_path = os.path.join(project_path, cur_index)

    # 2.2 Training conditions
    with open(config_path) as f:
        config = json.load(f)
    # Use GPU if possible.
    device = torch.device(
        config["gpu"]["device"] if torch.cuda.is_available() else "cpu"
    )
    n_node = config["data"]["n_node"]
    model_name = config["model"][
        "model_name"
    ]  # select from {vgg19_bn, mobilenet_v2, resnet_152, vit_b16}
    n_middle = config["model"]["n_middle"]
    useGPUinTrans = config["gpu"][
        "transform_on_GPU"
    ]  # whether use GPU in transform or not
    # self-training
    batch_size = config["collaborative_training"]["batch_size"]
    pretrain_epoch = config["self_training"]["epochs"]
    pretrain_lr = config["self_training"]["learning_rate"]
    pretrain_optimizer_name = config["self_training"]["optimizer_name"]  # SGD or Adam
    pretrain_momentum = config["self_training"]["momentum"]
    # collaborative training
    max_epoch = config["collaborative_training"]["epochs"]
    lr = config["collaborative_training"]["learning_rate"]
    fl_coefficient = config["collaborative_training"]["fl_coefficient"]
    optimizer_name = config["collaborative_training"]["optimizer_name"]  # SGD or Adam
    momentum = config["collaborative_training"]["momentum"]

    # 2.3 Schedulers (lower the learning rate during training)
    use_pretrain_scheduler = config["self_training"][
        "use_scheduler"
    ]  # whether to use scheduler in pretrain phase. Set to True if using lr_decay in pretrain phase.
    pretrain_scheduler_step = config["self_training"]["scheduler_step"]
    pretrain_scheduler_rate = config["self_training"]["scheduler_rate"]
    use_scheduler = config["collaborative_training"][
        "use_scheduler"
    ]  # if not using scheduler, set to False
    scheduler_step = config["collaborative_training"]["scheduler_step"]
    scheduler_rate = config["collaborative_training"]["scheduler_rate"]

    # 2.4 About the data each node has. Set 'is_use_noniid_filter' to 'True' if using non-IID filter to create data for each node.
    is_use_noniid_filter = config["non_IID_filter"]["use_noniid_filter"]
    filter_rate = config["non_IID_filter"]["filter_rate"]
    filter_seed = config["non_IID_filter"]["filter_seed"]

    # 2.5 About contact patterns
    contact_file = config["contact_pattern"]["contact_file"]
    contact_file_path = os.path.join(contact_pattern_path, contact_file)

    # 2.6 Select train mode
    is_pretrain_only = config["mode"]["self_train_only"]  # use to do only pre-training

    # 2.7 Settings
    torch_seed()  # fix the seed # Change the location------------
    g = torch.Generator()
    g.manual_seed(0)

    print("using device", device)

    ## 3. Test Transform (what processing to perform when calling data in each epoch. Train data will be processed later)
    # Load the mean and standard deviation of pixel values of test images in the Test directory for normalization
    if (os.path.exists(meant_file_path)) and (
        os.path.exists(stdt_file_path)
    ):  # If the files for mean and standard deviation of test data exist
        mean_t = torch.load(meant_file_path)
        std_t = torch.load(stdt_file_path)
    else:  # If the files for mean and standard deviation of test data do not exist
        mean_t, std_t = calculate_mean_and_std(test_path)
        torch.save(mean_t, meant_file_path)
        torch.save(std_t, stdt_file_path)
    print("calculation of mean and std in test data finished")

    if useGPUinTrans:  # If performing transform on GPU
        test_transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(
                    mean=tuple(mean_t.tolist()), std=tuple(std_t.tolist())
                ),
            ]
        )
    else:
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=tuple(mean_t.tolist()), std=tuple(std_t.tolist())
                ),
            ]
        )

    ## 4. Prepare data
    # 4.1 Create dataset with ImageFolder & custom dataset
    if useGPUinTrans:  # If performing transform on GPU
        # train_data = MyGPUdataset(train_path, device, pre_transform=transforms.Resize(256))
        train_data = ImageFolder(
            train_path
        )  # Train data will be normalized based on the mean and standard deviation of each node later
        test_data = MyGPUdataset(
            test_path,
            device,
            transform=test_transform,
            pre_transform=transforms.Resize(256),
        )
    else:
        train_data = datasets.ImageFolder(train_path)
        test_data = datasets.ImageFolder(test_path, transform=test_transform)

    # 4.2 loading filter file or not. Load Non-IID Filter
    filter_file = f"filter_r{filter_rate:02d}_s{filter_seed:02d}.pt"
    if is_use_noniid_filter:  # If using Non-IID Filter
        indices = torch.load(os.path.join(noniid_filter_path, filter_file))
    else:  # If not using Non-IID Filter
        indices = [[] for _ in range(n_node)]
        for i in range(len(train_data)):
            indices[i % n_node].append(i)

    # 4.3 Assign training data to each node
    subset = [Subset(train_data, indices[i]) for i in range(n_node)]
    # nums = [[0 for i in range(n_node)] for j in range(n_node)]
    # for i in range(n_node): # Output data distribution
    #     for j in range(len(subset[i])):
    #         image, label = subset[i][j]
    #         nums[i][int(label)] += 1
    #     print(f'Distributions of data')
    #     print(f"train_data of node_{i}: {nums[i]}\n")

    # Load files for normalization
    if is_use_noniid_filter:  # If using Non-IID filter
        train_mean_file_path = os.path.join(
            mean_and_std_path, f"mean_r{filter_rate:02d}_s{filter_seed:02d}.pt"
        )
        train_std_file_path = os.path.join(
            mean_and_std_path, f"std_r{filter_rate:02d}_s{filter_seed:02d}.pt"
        )
        if os.path.exists(train_mean_file_path) and os.path.exists(
            train_std_file_path
        ):  # If already calculated
            mean_list = torch.load(
                train_mean_file_path
            )  # Get the mean and standard deviation of pixel values for each node
            std_list = torch.load(
                train_std_file_path
            )  # Get the mean and standard deviation of pixel values for each node
        else:  # Calculate
            mean_list, std_list = calculate_mean_and_std_subset(subset)
            torch.save(mean_list, train_mean_file_path)
            torch.save(std_list, train_std_file_path)
    else:  # If not using Non-IID filter
        train_mean_file_path = os.path.join(mean_and_std_path, "IID_train_mean.pt")
        train_std_file_path = os.path.join(mean_and_std_path, "IID_train_std.pt")
        if os.path.exists(train_mean_file_path) and os.path.exists(train_std_file_path):
            mean_list = torch.load(train_mean_file_path)
            std_list = torch.load(train_std_file_path)
        else:
            mean_list, std_list = calculate_mean_and_std_subset(subset)
            torch.save(mean_list, train_mean_file_path)
            torch.save(std_list, train_std_file_path)
    print("Loading of mean and std in train data finished")

    # 4.4. Prepare train_dataloader
    trainloader = []
    for i in range(len(subset)):
        mean = mean_list[i]
        mean = mean.tolist()
        std = std_list[i]
        std = std.tolist()
        if useGPUinTrans:
            train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=224, scale=(0.4, 1.0)),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize(mean=tuple(mean), std=tuple(std)),
                    transforms.RandomErasing(
                        p=0.5,
                        scale=(0.02, 0.33),
                        ratio=(0.3, 3.3),
                        value=0,
                        inplace=False,
                    ),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=224, scale=(0.4, 1.0)),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize(mean=tuple(mean), std=tuple(std)),
                    transforms.RandomErasing(
                        p=0.5,
                        scale=(0.02, 0.33),
                        ratio=(0.3, 3.3),
                        value=0,
                        inplace=False,
                    ),
                ]
            )
        pre_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.uint8),
                # Store the original image as uint8 to compress memory capacity
                transforms.Resize(256),
            ]
        )
        # Perform common processing for all epochs using pre_transform
        train_dataset_new = FromSubsetDataset(
            subset[i],
            device,
            transform=train_transform,
            pre_transform=pre_transform,
            useGPUinTrans=useGPUinTrans,
        )
        # Specify different processing for each epoch, such as randomCrop, in the transform.
        train_dataset_new = FromSubsetDataset(
            subset[i],
            device,
            transform=train_transform,
            pre_transform=pre_transform,
            useGPUinTrans=useGPUinTrans,
        )
        if useGPUinTrans:
            trainloader.append(
                DataLoader(
                    train_dataset_new,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    worker_init_fn=seed_worker,
                    generator=g,
                )
            )
        else:
            trainloader.append(
                DataLoader(
                    train_dataset_new,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=50,
                    pin_memory=True,
                    worker_init_fn=seed_worker,
                    generator=g,
                )
            )

    # 4.5 Prepare test_dataloader
    if useGPUinTrans:
        testloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        testloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=50,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    ## 5. define net, optimizer, criterion
    # 5.1 Define the loss function
    criterion = nn.CrossEntropyLoss()

    # 5.2 Define the model
    nets = [
        select_net(model_name, len(classes), n_middle).to(device) for i in range(n_node)
    ]  # Get the model based on the model_name.

    # 5.3 Define the optimization function
    # Optimizer for model ensemble
    optimizers = [
        select_optimizer(model_name, nets[i], optimizer_name, lr, momentum)
        for i in range(n_node)
    ]
    pretrain_optimizers = [  # Optimizers for pretraining
        select_optimizer(
            model_name, nets[i], pretrain_optimizer_name, pretrain_lr, pretrain_momentum
        )
        for i in range(n_node)
    ]

    schedulers = None
    # If using lr decay in training
    if use_scheduler:
        schedulers = [
            optim.lr_scheduler.StepLR(
                optimizers[i], step_size=scheduler_step, gamma=scheduler_rate
            )
            for i in range(10)
        ]

    pretrain_schedulers = None
    # If using lr decay in pretraining
    if use_pretrain_scheduler:
        pretrain_schedulers = [
            optim.lr_scheduler.StepLR(
                pretrain_optimizers[i],
                step_size=pretrain_scheduler_step,
                gamma=pretrain_scheduler_rate,
            )
            for i in range(10)
        ]

    ## 6. Training Phase
    contact_list = []
    histories = [
        np.zeros((0, 5)) for i in range(n_node)
    ]  # Store the results of model training
    pretrain_histories = [
        np.zeros((0, 5)) for i in range(n_node)
    ]  # Store the results of pretraining

    ## 7. Main Loop
    os.makedirs(cur_path)  # Make a directory to store the execution results.
    show_dataset_contents(data_path, classes, cur_path)

    with open(
        os.path.join(cur_path, "log.txt"), "a"
    ) as f:  # Write the training settings to the log file
        for i in range(len(subset)):
            f.write(f"the number of data for training {i}-th node: {len(subset[i])}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"contact file: {contact_file}\n")
        if is_use_noniid_filter:
            f.write(f"filter file: {filter_file}\n")
        else:
            f.write("do not use filter\n")
        f.write(f"test transform: {testloader.dataset.transform}\n")
        f.write(f"train transform: {trainloader[0].dataset.transform}\n")
        f.write(f"optimizer: {optimizers[0]}\n")
        if use_scheduler:
            f.write(f"training scheduler: {schedulers[0].state_dict()}\n")
        f.write(f"pre-training optimizer: {pretrain_optimizers[0]}\n")
        if use_pretrain_scheduler:
            f.write(f"pre-training scheduler: {pretrain_schedulers[0].state_dict()}\n")
        f.write(f"net:\n {summary(nets[0], (1,3,224,224), verbose=False)}\n")

    os.makedirs(os.path.join(cur_path, "params"))
    load_epoch = 0
    # pre-self training
    pretrain(
        nets,
        trainloader,
        testloader,
        pretrain_optimizers,
        criterion,
        pretrain_epoch,
        device,
        cur_path,
        histories,
        pretrain_histories,
        pretrain_schedulers,
    )

    history_save_path = os.path.join(cur_path, "params", "histories_data.pkl")
    with open(history_save_path, "wb") as f:
        pickle.dump(histories, f)
    print("saving histories...")

    # exit if pretraining only
    if is_pretrain_only:
        pretrain_history_save_path = os.path.join(
            cur_path, "params", "pretrain_histories_data.pkl"
        )
        with open(
            pretrain_history_save_path, "wb"
        ) as f:  # Save the results of pretraining
            pickle.dump(pretrain_histories, f)
            print("saving pretrain histories...")
        exit(0)

    # load contact pattern
    print(f"Loading ... {contact_file_path}")
    with open(contact_file_path) as f:
        contact_list = json.load(f)

    for epoch in range(load_epoch, max_epoch + load_epoch):
        contact = contact_list[epoch]

        model_exchange(
            nets, model_name, contact, fl_coefficient
        )  # Perform model aggregation with the contacted nodes.

        for n in range(n_node):  # Train one epoch if the model was aggregated
            nbr = contact[str(n)]
            if len(nbr) == 0:
                item = np.array(
                    [
                        epoch + 1,
                        histories[n][-1][1],
                        histories[n][-1][2],
                        histories[n][-1][3],
                        histories[n][-1][4],
                    ]
                )
                histories[n] = np.vstack((histories[n], item))
                print(
                    f"Epoch [{epoch+1}], Node [{n}], loss: {histories[n][-1][1]:.5f} acc: {histories[n][-1][2]:.5f} val_loss: {histories[n][-1][3]:.5f} val_acc: {histories[n][-1][4]:.5f}"
                )
            else:
                histories[n] = fit(
                    nets[n],
                    optimizers[n],
                    criterion,
                    trainloader[n],
                    testloader,
                    device,
                    histories[n],
                    epoch,
                    n,
                )

            # Output confusion matrix
            if (
                (load_epoch + epoch > (load_epoch + max_epoch) * 0.75)
                and epoch % 50 == 49
            ) or (load_epoch + max_epoch - epoch < 11):
                train_for_cmls(
                    cur_path,
                    epoch,
                    n,
                    cur_index,
                    classes,
                    nets[n],
                    criterion,
                    testloader,
                    device,
                )

            # write log
            if epoch % 100 == 99:
                with open(os.path.join(cur_path, "log.txt"), "a") as f:
                    f.write(
                        f"Epoch [{epoch+1}], Node [{n}], loss: {histories[n][-1][1]:.5f} acc: {histories[n][-1][2]:.5f} val_loss: {histories[n][-1][3]:.5f} val_acc: {histories[n][-1][4]:.5f}\n"
                    )

            # save models
            if epoch == max_epoch + load_epoch - 1:
                print(f"Model saving ... at {epoch+1}")
                torch.save(
                    nets[n].state_dict(),
                    os.path.join(cur_path, f"params/node{n}_epoch-{epoch+1:04d}.pth"),
                )
                nets[n] = nets[n].to(device)

            # update scheduler
            if schedulers != None:
                schedulers[n].step()
                print(schedulers[n].get_last_lr())

    history_save_path = os.path.join(cur_path, "params", "histories_data.pkl")
    history_csv_save_path = os.path.join(cur_path, "params", "histories_data.csv")
    with open(history_save_path, "wb") as f:
        pickle.dump(histories, f)
        print("saving histories...")
    mean, std = calc_res_mean_and_std(histories)
    with open(os.path.join(cur_path, "log.txt"), "a") as f:
        f.write(f"the average of the last 10 epoch: {mean}\n")
        f.write(f"the std of the last 10 epoch: {std}\n")
    evaluate_history(histories, cur_path)
    print("Finished Training")
