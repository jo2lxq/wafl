import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--preself', type=bool, default=False)

args = parser.parse_args()

def read_mAP_data_from_csv(filename):
    mAPs = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        
        for i, row in enumerate(reader):
            if i == 0: 
                continue
            mAPs.append(float(row[6]))
    
    return mAPs


if __name__ == '__main__':
    #ファイル名を指定してlossデータを読み込み
    args = parser.parse_args()
    plt.figure(figsize=(10, 6))
    all_node_mAPs = []
    for i in range(6):
        filepath = f'outputs/node{i}/results.csv'
        if args.preself:
            filepath = f'outputs/preself/node{i}/results'
        mAPs = read_mAP_data_from_csv(filepath)
        epochs = range(1, len(mAPs) + 1)
        plt.plot(epochs, mAPs, label=f'node{i}')
        all_node_mAPs.append(mAPs)

    #全ノードのmAP描画
    plt.title('mAP@IoU=0.5')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    plt.savefig("mAP.png")
    plt.close()

    #ノードごとの平均mAPの描画
    plt.figure(figsize=(10, 6))
    avg_mAPs = np.average(np.array(all_node_mAPs), axis=0).tolist()
    epochs = range(len(avg_mAPs))
    plt.plot(epochs, avg_mAPs)
    plt.title('mAP@IoU=0.5')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.grid(True)
    plt.savefig("mAP_avg.png")
    plt.close()