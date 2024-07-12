import math
import os
import random
import json

n_time = 10000

n_node = 10
min_travel_speed = 3
max_travel_speed = 7
radio_range = 100

pose_time_set = [10, 40, 100]
areasize_set = [500]
randomseed_set = [1]

current_path = os.path.dirname(os.path.abspath(__file__))
contact_pattern_dir = os.path.normpath(os.path.join(current_path, '../../data/contact_pattern'))

parameters = []
for areasize in areasize_set:
    for pose_time in pose_time_set:
        for randomseed in randomseed_set:
            parameters.append((areasize, pose_time, randomseed))

for areasize, pose_time, randomseed in parameters:
    area = (areasize, areasize)
    random.seed(randomseed)

    node_location = [None] * n_node
    node_travel_speed = [None] * n_node
    node_pose_remaining_time = [pose_time] * n_node
    node_next_location = [None] * n_node

    for i in range(n_node):
        max_x, max_y = area
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        node_location[i] = (x, y)

    contact_list = []
    for t in range(n_time):
        for i in range(n_node):
            if node_pose_remaining_time[i] == 0:
                x, y = node_location[i]
                tx, ty = node_next_location[i]
                ax = tx - x
                ay = ty - y
                vx = node_travel_speed[i] * ax / math.sqrt(ax**2 + ay**2)
                vy = (
                    node_travel_speed[i] * ay / math.sqrt(ax**2 + ay**2)
                )  # 速度の成分を求める
                x += vx
                y += vy

                if (x - tx) ** 2 + (y - ty) ** 2 < node_travel_speed[i] ** 2:
                    node_location[i] = tx, ty
                    node_travel_speed[i] = None
                    node_next_location[i] = None
                    node_pose_remaining_time[i] = pose_time

                else:
                    node_location[i] = x, y

            else:
                node_pose_remaining_time[i] -= 1
                if node_pose_remaining_time[i] == 0:
                    max_x, max_y = area
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)
                    node_next_location[i] = (x, y)
                    node_travel_speed[i] = random.uniform(
                        min_travel_speed, max_travel_speed
                    )

        node_in_contact = {
            i: [] for i in range(n_node)
        }  # ある時刻においてノードiと通信可能なすべてのノードの番号をリストの中に入れている
        for i in range(n_node):
            node_in_contact[i] = []

            for j in range(n_node):
                if i != j:
                    xi, yi = node_location[i]
                    xj, yj = node_location[j]

                    if (xi - xj) ** 2 + (yi - yj) ** 2 < radio_range**2:
                        node_in_contact[i].append(j)

        print(f"t={t} : contacts={node_in_contact}")
        contact_list.append(node_in_contact)

    with open(
        os.path.join(
            contact_pattern_dir,
            f"rwp_n{n_node:02d}_a{areasize:04d}_r{radio_range:03d}_p{pose_time:02d}_s{randomseed:02d}.json",
        ),
        "w",
    ) as f:
        json.dump(contact_list, f, indent=4)
