import itertools
import os
import statistics

# import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import pickle
import sys
import os
from matplotlib.colors import ListedColormap

# 70個の色を定義
colors = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#000000", "#FFFFFF",
    "#800000", "#008000", "#000080", "#808000", "#800080", "#008080", "#808080", "#C00000",
    "#00C000", "#0000C0", "#C0C000", "#C000C0", "#00C0C0", "#C0C0C0", "#400000", "#004000",
    "#000040", "#404000", "#400040", "#004040", "#404040", "#200000", "#002000", "#000020",
    "#202000", "#200020", "#002020", "#202020", "#600000", "#006000", "#000060", "#606000",
    "#600060", "#006060", "#606060", "#A00000", "#00A000", "#0000A0", "#A0A000", "#A000A0",
    "#00A0A0", "#A0A0A0", "#E00000", "#00E000", "#0000E0", "#E0E000", "#E000E0", "#00E0E0",
    "#E0E0E0", "#B00000", "#00B000", "#0000B0", "#B0B000", "#B000B0", "#00B0B0", 
    "#D00000", "#00D000"
]


def evaluate_history(histories, cur_dir):
    cmap_name = "tab20"
    cm = plt.colormaps[cmap_name]
    with open(os.path.join(cur_dir, "log.txt"), "a") as f:
        f.write(f"Training until {len(histories[0])} epoch\n")
        for i in range(len(histories)):
            f.write(
                f"Initial Epoch (node{i}): Loss: {histories[i][0, 3]:.5f} Accuracy: {histories[i][0, 4]:.5f}\n"
            )
            f.write(
                f"Final Epoch (node{i}): Loss: {histories[i][-1, 3]:.5f} Accuracy: {histories[i][-1, 4]:.5f}\n"
            )
    img_dir_path = os.path.join(cur_dir, "images")
    if not (os.path.exists(img_dir_path)) or os.path.isfile(img_dir_path):
        os.makedirs(img_dir_path)
    num_epochs = len(histories[0])
    unit = num_epochs / 10

    plt.figure(figsize=(9, 8))
    for i in range(len(histories)):
        index_even = (i * 2) % 20
        index_odd = (i * 2 + 1) % 20
        print(cm.colors[index_even])
        plt.plot(
            histories[i][:, 0],
            histories[i][:, 1],
            label=f"node{i}(Train)",
            linewidth=0.5,
            color=cm.colors[index_even],
        )
        plt.plot(
            histories[i][:, 0],
            histories[i][:, 3],
            label=f"node{i}(Test)",
            linewidth=0.5,
            color=cm.colors[index_odd],
        )
    plt.xticks(np.arange(0, num_epochs, unit))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning curve (loss)")
    plt.legend(ncol=2)
    plt.savefig(os.path.join(cur_dir, "images/loss.png"))

    plt.figure(figsize=(9, 8))
    for i in range(len(histories)):
        index_even = (i * 2) % 20
        index_odd = (i * 2 + 1) % 20
        plt.plot(
            histories[i][:, 0],
            histories[i][:, 2],
            label=f"node{i}(Train)",
            linewidth=0.5,
            color=cm.colors[index_even],
        )
        plt.plot(
            histories[i][:, 0],
            histories[i][:, 4],
            label=f"node{i}(Test)",
            linewidth=0.5,
            color=cm.colors[index_odd],
        )
    plt.xticks(np.arange(0, num_epochs + 1, unit))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning curve (accuracy)")
    plt.legend(ncol=2)
    plt.savefig(os.path.join(cur_dir, "images/acc.png"))


def show_image_labels(loader, classes, net, device, loaded_epoch, cur_dir, all_images):
    if all_images:
        plt.figure(figsize=(20, 100))
        row_num = (
            len(loader.dataset) // 10
            if len(loader.dataset) % 10 == 0
            else len(loader.dataset) // 10 + 1
        )
    else:
        plt.figure(figsize=(20, 15))
        row_num = (
            loader.batch_size // 10
            if loader.batch_size % 10 == 0
            else loader.batch_size // 10 + 1
        )
    index = 0
    for images, labels in loader:
        n_size = len(images)
        if net is not None:
            if images.device == "cpu":
                images = images.to(device)
                # labels = labels.to(device)
            net.eval()
            outputs = net(images)
            predicted = torch.max(outputs, 1)[1]
            images = images.to("cpu")

        for i in range(n_size):
            ax = plt.subplot(row_num, 10, index * n_size + i + 1)
            label_name = classes[labels[i]]
            if net is not None:
                predicted_name = classes[predicted[i]]
                if label_name == predicted_name:
                    c = "k"
                else:
                    c = "b"
                ax.set_title(label_name + ":" + predicted_name, c=c, fontsize=20)
            else:
                ax.set_title(label_name, fontsize=20)

            image_np = images[i].numpy().copy()
            img = np.transpose(image_np, (1, 2, 0))
            img = (img + 1) / 2
            plt.imshow(img)
            ax.set_axis_off()
        if not all_images:
            break
        index += 1
    plt.savefig(
        os.path.join(cur_dir, f"images/sample_images/si-epoch-{loaded_epoch}.png")
    )


def save_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title=f"Confusion matrix",
    cmap=plt.cm.Blues,
    save_path=None,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    # print(cm)

    plt.clf()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches="tight")


def make_latent_space(y_tests, y_outputs, epoch, ls_path, cur_node):
    reducer = umap.UMAP(random_state=42)
    reducer.fit(y_outputs)
    embedding = reducer.transform(y_outputs)

    # plot
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y_tests, cmap="Spectral", s=5)
    plt.gca().set_aspect("equal", "datalim")
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.title(
        f"UMAP projection of the output features @ epoch={epoch:d}, node={cur_node}",
        fontsize=12,
    )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(ls_path, bbox_inches="tight")

    # plot
    plt.clf()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y_tests, cmap="Spectral", s=5)
    plt.gca().set_aspect("equal", "datalim")
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.title(
        f"UMAP projection of features @ epoch={epoch:d}, node={cur_node}", fontsize=12
    )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(ls_path, bbox_inches="tight")


def calc_res_mean_and_std(histories):
    raw_data = []
    for i in range(len(histories)):
        for j in range(10):  # last 10 epochs
            raw_data.append(histories[i][-j - 1][4])
    mean = statistics.mean(raw_data)
    std = statistics.stdev(raw_data)
    return mean, std

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <histories_path> <cur_path>")
        sys.exit(1)

    histories_path = sys.argv[1]
    cur_path = os.path.split(os.path.dirname(histories_path))[0]

    if not os.path.exists(histories_path):
        print(f"Error: {histories_path} does not exist.")
        sys.exit(1)

    if not os.path.exists(cur_path):
        print(f"Error: {cur_path} does not exist.")
        sys.exit(1)

    with open(histories_path, "rb") as f:
        histories = pickle.load(f)

    evaluate_history(histories, cur_path)