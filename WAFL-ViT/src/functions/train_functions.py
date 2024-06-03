import itertools
import os
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.io as io
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

from .visualize import *

# デフォルトフォントサイズ変更
# plt.rcParams['font.size'] = 14
# # デフォルトグラフサイズ変更
# plt.rcParams['figure.figsize'] = (6,6)
# # デフォルトで方眼表示ON
# plt.rcParams['axes.grid'] = True
# np.set_printoptions(suppress=True, precision=5)

class MyGPUdatasetFolder(
    datasets.DatasetFolder
):  # use when put data on GPU in __getitem__
    IMG_EXTENTIONS = [".jpg", ".jpeg", ".png"]

    def __init__(self, root, device, transform=None):
        super().__init__(
            root, loader=self.custom_loader, extensions=self.IMG_EXTENTIONS
        )
        self.transform = transform
        self.device = device

    def custom_loader(self, path):
        return io.read_image(path) / 255.0

    def __getitem__(self, index):
        data, label = super().__getitem__(index)
        data = data.to(self.device)
        label = (torch.tensor(label)).to(self.device)
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.samples)

def pretrain(
    nets,
    train_loaders,
    test_loader,
    optimizers,
    criterion,
    num_epoch,
    device,
    cur_dir,
    histories,
    pretrain_histories,
    schedulers,
):
    for epoch in range(num_epoch):
        for n in range(len(train_loaders)):
            nets[n].train()
            data_train_num = 0
            n_train_acc, n_val_acc = 0, 0
            train_loss, val_loss = 0, 0
            for data in train_loaders[n]:
                # get the inputs; data is a list of [x_train, y_train]
                x_train, y_train = data
                batch_size = len(y_train)
                if x_train.device == "cpu":
                    x_train = x_train.to(device)
                    y_train = y_train.to(device)
                optimizers[n].zero_grad()
                y_output = nets[n](x_train)
                loss = criterion(y_output, y_train)
                loss.backward()
                optimizers[n].step()
                predicted = torch.max(y_output, 1)[1]
                n_train_acc += (predicted == y_train).sum().item()
                data_train_num += batch_size
                train_loss += loss.item() * batch_size

            nets[n].eval()
            data_test_num = 0
            for tdata in test_loader:
                x_test, y_test = tdata
                batch_size = len(y_test)
                if x_test.device == "cpu":
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
                y_output_t = nets[n](x_test)
                loss_t = criterion(y_output_t, y_test)
                predicted_t = torch.max(y_output_t, 1)[1]
                n_val_acc += (predicted_t == y_test).sum().item()
                data_test_num += batch_size
                val_loss += loss_t.item() * batch_size

            train_acc = n_train_acc / data_train_num
            val_acc = n_val_acc / data_test_num
            avg_train_loss = train_loss / data_train_num
            avg_val_loss = val_loss / data_test_num

            if epoch % 5 == 4:
                print(
                    f"Pre-self training: [{n}th-node, {epoch + 1}th-epoch] train_acc: {train_acc:.5f}, val_acc: {val_acc:.5f}"
                )
            if epoch % (num_epoch // 2) == (num_epoch // 2) - 1:
                with open(os.path.join(cur_dir, "log.txt"), "a") as f:
                    f.write(
                        f"Pre-self training: [{n}th-node, {epoch + 1}th-epoch] train_acc: {train_acc:.5f}, val_acc: {val_acc:.5f}\n"
                    )
            item = np.array([0, avg_train_loss, train_acc, avg_val_loss, val_acc])
            item_p = np.array(
                [epoch + 1, avg_train_loss, train_acc, avg_val_loss, val_acc]
            )
            pretrain_histories[n] = np.vstack((pretrain_histories[n], item_p))
            if epoch == num_epoch - 1:
                histories[n] = np.vstack((histories[n], item))
            if schedulers != None:
                schedulers[n].step()
                print(schedulers[n].get_last_lr())

    for n in range(len(train_loaders)):
        torch.save(
            nets[n].state_dict(), os.path.join(cur_dir, f"params/Pre-train-node{n}.pth")
        )

def fit(
    net,
    optimizer,
    criterion,
    train_loader,
    test_loader,
    device,
    history,
    cur_epoch,
    cur_node,
):
    n_train_acc, n_val_acc = 0, 0
    train_loss, val_loss = 0, 0
    n_train, n_test = 0, 0

    net.train()
    for inputs, labels in train_loader:
        train_batch_size = len(labels)
        n_train += train_batch_size
        if inputs.device == "cpu":
            inputs = inputs.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predicted = torch.max(outputs, 1)[1]
        train_loss += loss.item() * train_batch_size
        n_train_acc += (predicted == labels).sum().item()

    net.eval()
    for inputs_test, labels_test in test_loader:
        test_batch_size = len(labels_test)
        n_test += test_batch_size
        if inputs_test.device == "cpu":
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

        outputs_test = net(inputs_test)
        loss_test = criterion(outputs_test, labels_test)

        predicted_test = torch.max(outputs_test, 1)[1]
        val_loss += loss_test.item() * test_batch_size
        n_val_acc += (predicted_test == labels_test).sum().item()

    train_acc = n_train_acc / n_train
    val_acc = n_val_acc / n_test
    avg_train_loss = train_loss / n_train
    avg_val_loss = val_loss / n_test
    print(
        f"Epoch [{cur_epoch+1}], Node [{cur_node}], loss: {avg_train_loss:.5f} acc: {train_acc:.5f} val_loss: {avg_val_loss:.5f} val_acc: {val_acc:.5f}"
    )
    item = np.array([cur_epoch + 1, avg_train_loss, train_acc, avg_val_loss, val_acc])
    history = np.vstack((history, item))
    return history

def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def show_dataset_contents(
    data_path, classes, result_path
):  # classes: ("label1", "label2", ...), print dataset divided by classes
    num_test = [0 for i in range(len(classes))]
    num_train = [0 for i in range(len(classes))]
    for i, label in enumerate(classes):
        class_path_train = os.path.join(data_path, "train", str(i))
        files_and_dirs_in_container_train = os.listdir(class_path_train)
        files_list_train = [
            d
            for d in files_and_dirs_in_container_train
            if os.path.isfile(os.path.join(class_path_train, d))
        ]
        num_train[i] = len(files_list_train)

        class_path_test = os.path.join(data_path, "val", str(i))
        files_and_dirs_in_container_test = os.listdir(class_path_test)
        files_list_test = [
            d
            for d in files_and_dirs_in_container_test
            if os.path.isfile(os.path.join(class_path_test, d))
        ]
        num_test[i] = len(files_list_test)

    with open(os.path.join(result_path, "log.txt"), "w") as f:
        for i in range(len(num_test)):
            # print(f'label: {classes[i]} train_data: {num_train[i]} test_data: {num_test[i]}')
            f.write(
                f"label: {classes[i]} train_data: {num_train[i]} test_data: {num_test[i]}\n"
            )

def calculate_mean_and_std(datapath):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(datapath, transform=transform)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = len(dataset)

    for i in range(total_samples):
        image, _ = dataset[i]
        # print(image.mean())
        mean += image.mean(dim=(1, 2))
        std += image.std(dim=(1, 2))

    mean /= total_samples
    std /= total_samples
    return mean, std

def calculate_mean_and_std_subset(subset):
    mean_list = [torch.zeros(3) for _ in range(len(subset))]
    std_list = [torch.zeros(3) for _ in range(len(subset))]
    for i in range(len(subset)):
        total_samples = len(subset[i])
        for j in range(total_samples):
            image, _ = subset[i][j]
            image = transforms.ToTensor()(image)
            mean_list[i] += image.mean(dim=(1, 2))
            std_list[i] += image.std(dim=(1, 2))
        mean_list[i] /= total_samples
        std_list[i] /= total_samples
    return mean_list, std_list

def train_for_cmls(
    cur_dir, epoch, n, cur_time_index, classes, net, criterion, test_loader, device
):
    n_val_acc = 0
    val_loss = 0
    n_test = 0
    y_preds = []  # for confusion_matrix
    y_tests = []
    y_outputs = []
    net.eval()
    for inputs_test, labels_test in test_loader:
        test_batch_size = len(labels_test)
        n_test += test_batch_size
        if inputs_test.device == "cpu":
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

        outputs_test = net(inputs_test)
        # outputs_test2 = tmp_net(inputs_test)
        loss_test = criterion(outputs_test, labels_test)

        predicted_test = torch.max(outputs_test, 1)[1]
        y_preds.extend(predicted_test.tolist())
        y_tests.extend(labels_test.tolist())
        y_outputs.extend(outputs_test.tolist())
        # z_outputs.extend(outputs_test2.tolist())
        val_loss += loss_test.item() * test_batch_size
        n_val_acc += (predicted_test == labels_test).sum().item()

    # make confusion matrix
    # cm_dir_path = os.path.join(cur_dir, "images/confusion_matrix")
    # if not (os.path.exists(cm_dir_path)) or os.path.isfile(cm_dir_path):
    #     os.makedirs(cm_dir_path)
    normalized_cm_dir_path = os.path.join(cur_dir, "images/normalized_confusion_matrix")
    if not (os.path.exists(normalized_cm_dir_path)) or os.path.isfile(
        normalized_cm_dir_path
    ):
        os.makedirs(normalized_cm_dir_path)
    confusion_mtx = confusion_matrix(y_tests, y_preds)
    # save_confusion_matrix(
    #     confusion_mtx,
    #     classes=classes,
    #     normalize=False,
    #     title=f"Confusion Matrix in {cur_time_index} at {epoch+1:d}epoch (node{n})",
    #     cmap=plt.cm.Reds,
    #     save_path=os.path.join(
    #         cur_dir, f"images/confusion_matrix/cm-epoch-{epoch+1:04d}-node{n}.png"
    #     ),
    # )
    print("Saving confusion matrix...")
    save_confusion_matrix(
        confusion_mtx,
        classes=classes,
        normalize=True,
        title=f"Normalized Confusion Matrix in {cur_time_index} at {epoch+1:d}epoch (node{n})",
        cmap=plt.cm.Reds,
        save_path=os.path.join(
            cur_dir,
            f"images/normalized_confusion_matrix/normalized-cm-epoch{epoch+1:04d}-node{n}.png",
        ),
    )

    # make latent space
    ls_dir_path = os.path.join(cur_dir, "images/latent_space")
    if not (os.path.exists(ls_dir_path)) or os.path.isfile(ls_dir_path):
        os.makedirs(ls_dir_path)
    make_latent_space(
        y_tests,
        y_outputs,
        epoch + 1,
        os.path.join(ls_dir_path, f"ls-epoch{epoch+1:4d}-node{n}.png"),
        n,
    )

def select_optimizer(model_name, net, optimizer_name, lr, momentum=None):
    if optimizer_name == "SGD":
        if model_name == "vgg19_bn":
            optimizer = optim.SGD(
                net.classifier[6].parameters(), lr=lr, momentum=momentum
            )
        elif model_name == "resnet_152":
            optimizer = optim.SGD(net.fc.parameters(), lr=lr, momentum=momentum)
        elif model_name == "mobilenet_v2":
            optimizer = optim.SGD(
                net.classifier[1].parameters(), lr=lr, momentum=momentum
            )
        elif model_name == "vit_b16":
            optimizer = optim.SGD(net.heads.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "Adam":
        if model_name == "vgg19_bn":
            optimizer = optim.Adam(net.classifier[6].parameters(), lr=lr)
        elif model_name == "resnet_152":
            optimizer = optim.Adam(net.fc.parameters(), lr=lr)
        elif model_name == "mobilenet_v2":
            optimizer = optim.Adam(net.classifier[1].parameters(), lr=lr)
        elif model_name == "vit_b16":
            optimizer = optim.Adam(net.heads.parameters(), lr=lr)
    return optimizer