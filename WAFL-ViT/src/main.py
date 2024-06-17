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
    ## 1. 初期設定とPath
    # Use GPU if possible.
    device = torch.device(
        "cuda:1" if torch.cuda.is_available() else "cpu"
    )  # use 0 in GPU1 use 1 in GPU2
    # path
    main_path = os.path.dirname(os.path.abspath(__file__)) # Absolute path to main.py. Note that "main_path" does not include file name i.e. "main.py".
    data_path = os.path.normpath(os.path.join(main_path, "../data"))
    project_path = os.path.normpath(os.path.join(main_path, "../training_logs"))
    noniid_filter_path = os.path.normpath(os.path.join(main_path, "../data/non-IID_filter"))
    contact_pattern_path = os.path.normpath(os.path.join(main_path, "../data/contact_pattern"))
    mean_and_std_path = os.path.normpath(os.path.join(main_path, "../data/test_mean_and_std"))
    # classes = ("安田講堂", "工2", "工3", "工13", "工4", "工8", "工1", "工6", "列品館", "法文1")
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "val")
    meant_file_path = os.path.join(
        mean_and_std_path, "test_mean.pt"
    )  ## IIDでもNonIIDでもtest用は同じ
    stdt_file_path = os.path.join(
        mean_and_std_path, "test_std.pt"
    )  ## IIDでもNonIIDでもtest用は同じ

    ## 2. 変数の定義（ここを変更する）
    # 2.1 出力ファイル名
    cur_index = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # cur_index = "Sample"
    cur_path = os.path.join(project_path, cur_index)

    # 2.2 Training conditions
    max_epoch = 3000
    pretrain_epoch = 100
    batch_size = 16
    n_node = 10
    n_middle = 256
    fl_coefficiency = 0.1
    model_name = "mobilenet_v2"  # select from {vgg19_bn, mobilenet_v2, resnet_152, vit_b16}
    optimizer_name = "SGD"  # SGD or Adam
    useGPUinTrans = True  # whether use GPU in transform or not
    lr = 0.01
    momentum = 0.9
    pretrain_lr = 0.01
    pretrain_momentum = 0.9

    # 2.3 Schedulers（学習率を途中で下げる）
    use_scheduler = False  # if do not use scheduler, False here
    scheduler_step = 10
    scheduler_rate = 0.5
    use_pretrain_scheduler = False  # pretrainでschedulerを使うか. When you use lr_decay in pretrain phase, set True.
    pretrain_scheduler_step = 10
    pretrain_scheduler_rate = 0.3

    # 2.4 About the data each node has. When you use non-IID filter to create data of each node, 'is_use_noniid_filter' is 'True'.
    is_use_noniid_filter = False
    filter_rate = 50
    filter_seed = 1

    # 2.5 About contact patterns
    # contact_file=f'cse_n10_c10_b02_tt05_tp2_s01.json'
    # contact_file = 'meet_at_once_t10000.json'
    contact_file = "rwp_n10_a0500_r100_p10_s01.json"  ## 場所を変更する---------------
    contact_file_path = os.path.join(contact_pattern_path, contact_file)

    # 2.6 Select train mode
    is_pretrain_only = False  # use to do only pre-training

    # 2.7 設定
    torch_seed()  # seedの固定 # 場所の変更------------
    g = torch.Generator()
    g.manual_seed(0)

    print("using device", device)
    # schedulers = None ## 場所の変更---------- 実体生成用の変数宣言は一箇所（ここ？）にまとめる
    # pretrain_schedulers = None

    ## 3. Test Transform （各エポックでのデータ呼び出しの際に、RandomCropなどどのような処理を行うか. Train用は後で）
    # Normalize用に、Test directoryの画像の画素値の平均と分散を読み込む
    if (os.path.exists(meant_file_path)) and (
        os.path.exists(stdt_file_path)
    ):  # Test dataの平均と分散のファイルがある場合
        mean_t = torch.load(meant_file_path)
        std_t = torch.load(stdt_file_path)
    else:  # Test dataの平均と分散のファイルがない場合
        mean_t, std_t = calculate_mean_and_std(test_path)
        torch.save(mean_t, meant_file_path)
        torch.save(std_t, stdt_file_path)
    print("calculation of mean and std in test data finished")

    if useGPUinTrans:  # GPUでTransformを行う場合
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
    # train_data = None
    # test_data = None
    if useGPUinTrans:  # GPUでTransformを行う場合
        # train_data = MyGPUdataset(train_path, device, pre_transform=transforms.Resize(256))
        train_data = ImageFolder(
            train_path
        )  # Trainのデータは後でノードごとの平均、分散に基づく正規化を行う
        test_data = MyGPUdataset(
            test_path,
            device,
            transform=test_transform,
            pre_transform=transforms.Resize(256),
        )
    else:
        train_data = datasets.ImageFolder(train_path)
        test_data = datasets.ImageFolder(test_path, transform=test_transform)

    # 4.2 loading filter file or not. Non-IID Filterを読み込む
    filter_file = f"filter_r{filter_rate:02d}_s{filter_seed:02d}.pt"
    if is_use_noniid_filter:  # Non-IID Filterを使う場合
        indices = torch.load(os.path.join(noniid_filter_path, filter_file))
    else:  # Non-IID Filterを使わない場合
        indices = [[] for _ in range(n_node)]
        for i in range(len(train_data)):
            indices[i % n_node].append(i)

    # 4.3 Assign training data to each node
    # for i in range(n_node):# データ分布の出力
    #     print(f"node_{i}:{indices[i]}\n")
    subset = [Subset(train_data, indices[i]) for i in range(n_node)]
    # nums = [[0 for i in range(n_node)] for j in range(n_node)]
    # for i in range(n_node): # データ分布の出力を行う
    #     for j in range(len(subset[i])):
    #         image, label = subset[i][j]
    #         nums[i][int(label)] += 1
    #     print(f'Distributions of data')
    #     print(f"train_data of node_{i}: {nums[i]}\n")

    # Normalize用のファイル読み込み
    if is_use_noniid_filter:  # Non-IIDフィルタを使うとき
        train_mean_file_path = os.path.join(
            mean_and_std_path, f"mean_r{filter_rate:02d}_s{filter_seed:02d}.pt"
        )
        train_std_file_path = os.path.join(
            mean_and_std_path, f"std_r{filter_rate:02d}_s{filter_seed:02d}.pt"
        )
        if os.path.exists(train_mean_file_path) and os.path.exists(
            train_std_file_path
        ):  # 既に計算済みの場合
            mean_list = torch.load(
                train_mean_file_path
            )  # 各ノードごとの画素値の平均、分散を取得
            std_list = torch.load(
                train_std_file_path
            )  # 各ノードごとの画素値の平均、分散を取得
        else:  # 計算する
            mean_list, std_list = calculate_mean_and_std_subset(subset)
            torch.save(mean_list, train_mean_file_path)
            torch.save(std_list, train_std_file_path)
    else:  # Non-IIDフィルタを使わないとき
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
        # train_transform = None
        if useGPUinTrans:
            train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=224, scale=(0.4, 1.0)),
                    # transforms.RandomCrop(224),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Normalize(
                        mean=tuple(mean), std=tuple(std)
                    ),  # 各ノードの画素値の平均、分散をもとに正規化
                    # transforms.Normalize(0.5, 0.5)
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
                    # transforms.ToTensor(), # すでにpreトランスフォームでTensor化しているのでは？
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
                transforms.ConvertImageDtype(
                    torch.uint8
                ),  # メモリ容量を圧縮するためにuint8で保管（元の画像はuint8なので情報の損失なし）
                transforms.Resize(256),
            ]
        )
        # 最初に全エポックに共通の処理をpre_transformで実施。
        # randomCropなどそれぞれのエポックで異なる処理はtransformに記載。
        train_dataset_new = FromSubsetDataset(
            subset[i], device, transform=train_transform, pre_transform=pre_transform
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
    # testloader = None

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
    # 5.1 損失関数を定義
    criterion = nn.CrossEntropyLoss()

    # 5.2 Transfer learningに使用するmodelを定義
    nets = [
        select_net(model_name, len(classes), n_middle).to(device) for i in range(n_node)
    ]  # model_nameに応じたモデルをとってくる。

    # 5.3 最適化関数を定義
    optimizers = [  # model合成時のoptimizer
        select_optimizer(model_name, nets[i], optimizer_name, lr, momentum)
        for i in range(n_node)
    ]
    pretrain_optimizers = [  # pretrain時のoptimizer
        select_optimizer(
            model_name, nets[i], optimizer_name, pretrain_lr, pretrain_momentum
        )
        for i in range(n_node)
    ]

    schedulers = None  # 下の学習関数で引数として与えるので、実体を作る
    if use_scheduler:  # exchangeでlr decayを使う場合
        schedulers = [
            optim.lr_scheduler.StepLR(
                optimizers[i], step_size=scheduler_step, gamma=scheduler_rate
            )
            for i in range(10)
        ]
    pretrain_schedulers = None  # 下の学習関数で引数として与えるので、実体を作る
    if use_pretrain_scheduler:  # pretrainでlr decayを使う場合
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
    histories = [np.zeros((0, 5)) for i in range(n_node)]  # モデル合成時の結果を格納
    pretrain_histories = [
        np.zeros((0, 5)) for i in range(n_node)
    ]  # pretrainの結果を格納

    ## 7. Main Loop
    os.makedirs(
        cur_path
    )  # 実行結果の格納用ディレクトリを作成. makedirsを使うと、親ディレクトリも含めて一気に新規作成できる。
    show_dataset_contents(data_path, classes, cur_path)

    with open(os.path.join(cur_path, "log.txt"), "a") as f:  # パラメータの出力
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

    # Pretrainを行う
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

    if is_pretrain_only:  # pretrainのみの場合はここで終了
        pretrain_history_save_path = os.path.join(
            cur_path, "params", "pretrain_histories_data.pkl"
        )
        with open(pretrain_history_save_path, "wb") as f:  # pretrainのhistoryを保存
            pickle.dump(pretrain_histories, f)
            print("saving pretrain histories...")
        exit(0)

    # load contact pattern
    print(f"Loading ... {contact_file_path}")
    with open(contact_file_path) as f:
        contact_list = json.load(f)

    for epoch in range(
        load_epoch, max_epoch + load_epoch
    ):  # loop over the dataset multiple times
        # print(global_model['fc2.bias'][1])
        contact = contact_list[epoch]

        model_exchange(
            nets, model_name, contact, fl_coefficiency
        )  # model_nameで指定したモデルで接触したノードとモデル合成を行う。

        for n in range(n_node):  # モデルの合成を行った場合は1回学習を行う
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

            # このif文がTrueになったらConfusion Matrixを出力
            if (
                (load_epoch + epoch > (load_epoch + max_epoch) * 0.75)
                and epoch % 50 == 49
            ) or (load_epoch + max_epoch - epoch < 11):
                # print(load_epoch)
                # print(max_epoch)
                # print(epoch)
                # print(load_epoch + max_epoch - epoch)
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
