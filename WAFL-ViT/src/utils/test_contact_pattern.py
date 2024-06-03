import json
import os

# このファイルはgenerate.pyで作ったものから読み込むので事前にgenerateが必要で下記パラメータはそれに合わせる

n_time = 3000  # up to 10000

n_node = 12  # fixed
radio_range = 100  # fixed

pose_time = 10  # [10,20,40]
areasize = 500  # [500,1000,2000]
randomseed = 1  # [1,2,3]

contact_pattern_dir = "../../data/contact_pattern"

contact_list = []

filename = os.path.join(
    contact_pattern_dir,
    f"rwp_n{n_node:02d}_a{areasize:04d}_r{radio_range:03d}_p{pose_time:02d}_s{randomseed:02d}.json",
)
print(f"Loading ... {filename}")
with open(filename) as f:
    contact_list = json.load(f)

if len(contact_list) < n_time:
    print(
        f"Warning: n_time={n_time} is larger than available times{len(contact_list)}."
    )

for t in range(n_time):
    contact = contact_list[t]

    print(f"At t={t}")
    for n in range(n_node):
        nbr = contact[str(n)]
        for k in nbr:
            print(f"    {n}=>{k}")
