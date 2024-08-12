import json
import os

"""
This file uses and checks the file created by generate.py, so you need to run generate.py before running this file.
The following parameters should be consistent with generate.py.
"""

n_time = 3000

n_node = 10
radio_range = 100

pose_time = 10
areasize = 500
randomseed = 1

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
