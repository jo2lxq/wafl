import json
import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# このファイルはgenerate.pyで作ったものから読み込むので、事前にgenerateが必要で下記パラメータはそれに合わせる

max_time = 1000

n_node = 12
min_travel_speed = 3
max_travel_speed = 7
radio_range = 100

pose_time_set = [10, 40]
areasize_set = [500]
randomseed_set = [1]

contact_pattern_dir = "../../data/contact_pattern"

parameters = []
for areasize in areasize_set:
    for pose_time in pose_time_set:
        for randomseed in randomseed_set:
            parameters.append((areasize, pose_time, randomseed))

for areasize, pose_time, randomseed in parameters:
    contact_list = []
    edges = []

    with open(
        os.path.join(
            contact_pattern_dir,
            f"rwp_n{n_node:02d}_a{areasize:04d}_r{radio_range:03d}_p{pose_time:02d}_s{randomseed:02d}.json",
        )
    ) as f:
        contact_list = json.load(f)

    for t in range(max_time):
        contact = contact_list[t]

        for s, l in contact.items():
            for t in l:
                if int(s) < t:
                    edge = [s, str(t)]
                    if edge not in edges:
                        edges.append(edge)

    from_node = []
    to_node = []

    for e in edges:
        from_node.append(e[0])
        to_node.append(e[1])

    df = pd.DataFrame({"from": from_node, "to": to_node})

    plt.clf()
    G = nx.from_pandas_edgelist(df, "from", "to")
    nx.draw(
        G,
        with_labels=True,
        node_size=1500,
        node_color="skyblue",
        pos=nx.fruchterman_reingold_layout(G),
    )
    if not os.path.exists(os.path.join(contact_pattern_dir, "graph")):
        os.makedirs(os.path.join(contact_pattern_dir, "graph"))
    filename = os.path.join(
        contact_pattern_dir,
        f"graph/rwp_n{n_node:02d}_a{areasize:04d}_r{radio_range:03d}_p{pose_time:02d}_s{randomseed:02d}.png",
    )
    print(f"Generating ... {filename}")
    plt.savefig(filename)
    # plt.show()
