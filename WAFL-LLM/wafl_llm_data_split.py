"""
wafl_llm_data_split.py
Split the MultiWOZ 2.4 training data over the n nodes of a WAFL network.

Two split strategies are provided:

  - "noniid" : domain-based Non-IID split.  Each node is biased towards the
               dialogues of one particular domain, which mirrors a ubiquitous
               setting where every device sees a different kind of situation.
  - "iid"    : random split, so every node ends up with roughly the same
               domain distribution.  Used as a baseline.

Requires mwz24_data.py and model_registry.py in the same directory.
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

from model_registry import DEFAULT_MODEL, resolve_model
from mwz24_data import (
    CACHE_DIR,
    download_mwz24,
    extract_belief_state,
    dialogue_to_samples,
)

# The five dominant MultiWOZ 2.4 domains, in order of frequency.
MAIN_DOMAINS = ["restaurant", "hotel", "train", "attraction", "taxi"]


# ===========================
# Primary domain of a dialogue
# ===========================

def get_primary_domain(dialogue: dict) -> str:
    """
    Return the domain that appears most often in the belief states of the
    dialogue.  Returns "misc" if no domain appears at all.
    """
    dom_count = Counter()
    for turn in dialogue.get("log", []):
        metadata = turn.get("metadata", {})
        if not metadata:
            continue
        belief = extract_belief_state(metadata)
        for key in belief:
            dom = key.split("-")[0]
            dom_count[dom] += 1

    if not dom_count:
        return "misc"
    return dom_count.most_common(1)[0][0]


# ===========================
# Loading the train dialogues
# ===========================

def _load_train_dialogues(cache_dir: str):
    """Return the dialogues of the train split as {dialogue_id: dialogue}."""
    data_dir = download_mwz24(cache_dir)

    with open(os.path.join(data_dir, "data.json"), encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(data_dir, "valListFile.json")) as f:
        val_ids = {line.strip() for line in f if line.strip()}
    with open(os.path.join(data_dir, "testListFile.json")) as f:
        test_ids = {line.strip() for line in f if line.strip()}

    train_ids = set(data.keys()) - val_ids - test_ids
    return {did: data[did] for did in train_ids}


def _build_domain_node_map(n_device: int) -> dict:
    """
    Decide which nodes are responsible for which of the main domains.

    With the default n_device=10 and five main domains, every domain gets
    exactly two nodes.  Leftover nodes are handed out round-robin over the
    domains (which are ordered by frequency), and if there are fewer nodes
    than domains, a node takes responsibility for several domains.
    """
    domain_node_map = {dom: [] for dom in MAIN_DOMAINS}

    if n_device >= len(MAIN_DOMAINS):
        nodes_per_domain = n_device // len(MAIN_DOMAINS)
        node_id = 0
        for dom in MAIN_DOMAINS:
            domain_node_map[dom] = list(range(node_id,
                                              node_id + nodes_per_domain))
            node_id += nodes_per_domain
        # Hand the remaining nodes to the most frequent domains first.
        i = 0
        while node_id < n_device:
            domain_node_map[MAIN_DOMAINS[i % len(MAIN_DOMAINS)]].append(node_id)
            node_id += 1
            i += 1
    else:
        # Fewer nodes than domains: each node covers more than one domain.
        for i, dom in enumerate(MAIN_DOMAINS):
            domain_node_map[dom] = [i % n_device]

    return domain_node_map


# ===========================
# Non-IID split (domain based)
# ===========================

def split_noniid(dialect: str, n_device: int = 10, seed: int = 1,
                 cache_dir: str = CACHE_DIR) -> list:
    """
    Domain-based Non-IID split.

    Dialogues whose primary domain is one of the five main domains go to the
    nodes responsible for that domain.  Dialogues of any other domain
    (misc, hospital, bus, ...) are spread thinly over all nodes so that no
    dialogue is wasted.

    Returns node_samples, where node_samples[node_id] = [sample, ...].
    """
    random.seed(seed)
    dialogues = _load_train_dialogues(cache_dir)

    # Group the dialogues by their primary domain.
    domain_to_dialogues = defaultdict(list)
    for did, dlg in dialogues.items():
        primary = get_primary_domain(dlg)
        domain_to_dialogues[primary].append(did)

    domain_node_map = _build_domain_node_map(n_device)

    # Assign dialogues to nodes.
    node_dialogue_ids = defaultdict(list)
    for dom, dids in sorted(domain_to_dialogues.items()):
        dids = sorted(dids)          # sort first so the shuffle is reproducible
        random.shuffle(dids)
        if dom in domain_node_map:
            target_nodes = domain_node_map[dom]
        else:
            target_nodes = list(range(n_device))
        for i, did in enumerate(dids):
            node = target_nodes[i % len(target_nodes)]
            node_dialogue_ids[node].append(did)

    # Expand the dialogues of every node into turn-level samples.
    node_samples = [[] for _ in range(n_device)]
    for node in range(n_device):
        for did in node_dialogue_ids[node]:
            node_samples[node].extend(dialogue_to_samples(dialogues[did], dialect))

    _print_split_stats("Non-IID (domain-based)", node_samples,
                       dialogues, node_dialogue_ids)
    return node_samples


# ===========================
# IID split (random)
# ===========================

def split_iid(dialect: str, n_device: int = 10, seed: int = 1,
              cache_dir: str = CACHE_DIR) -> list:
    """
    Random split.  All dialogues are shuffled and dealt out evenly over the
    nodes, so every node sees roughly the same domain distribution.  Baseline.
    """
    random.seed(seed)
    dialogues = _load_train_dialogues(cache_dir)

    all_ids = sorted(dialogues.keys())
    random.shuffle(all_ids)

    node_dialogue_ids = defaultdict(list)
    for i, did in enumerate(all_ids):
        node_dialogue_ids[i % n_device].append(did)

    node_samples = [[] for _ in range(n_device)]
    for node in range(n_device):
        for did in node_dialogue_ids[node]:
            node_samples[node].extend(dialogue_to_samples(dialogues[did], dialect))

    _print_split_stats("IID (random)", node_samples,
                       dialogues, node_dialogue_ids)
    return node_samples


# ===========================
# Split statistics
# ===========================

def _print_split_stats(name, node_samples, dialogues, node_dialogue_ids):
    print(f"\n===== Data Split: {name} =====")
    for node in range(len(node_samples)):
        # Domain distribution of this node.
        dom_count = Counter()
        for did in node_dialogue_ids[node]:
            dom_count[get_primary_domain(dialogues[did])] += 1
        top = ", ".join(f"{d}:{c}" for d, c in dom_count.most_common(3))
        print(f"  node {node}: {len(node_samples[node]):5d} samples "
              f"({len(node_dialogue_ids[node])} dialogues) | top: {top}")
    total = sum(len(s) for s in node_samples)
    print(f"  total: {total} samples")


def get_split(split_type: str, dialect: str, n_device: int = 10,
              seed: int = 1, cache_dir: str = CACHE_DIR) -> list:
    """Run the requested split.  split_type is either 'noniid' or 'iid'."""
    if split_type == "noniid":
        return split_noniid(dialect, n_device, seed, cache_dir)
    elif split_type == "iid":
        return split_iid(dialect, n_device, seed, cache_dir)
    else:
        raise ValueError(f"Unknown split_type: {split_type}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspect the WAFL data split of MultiWOZ 2.4.")
    p.add_argument("--split-type", default="both",
                   choices=["noniid", "iid", "both"],
                   help="which split to print (default: both)")
    p.add_argument("--n-device", type=int, default=10,
                   help="number of WAFL nodes (default: 10)")
    p.add_argument("--seed", type=int, default=1,
                   help="random seed of the split (default: 1)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="preset alias or Hugging Face id; only selects the "
                        "prompt dialect, which does not affect the counts "
                        f"below (default: {DEFAULT_MODEL})")
    p.add_argument("--cache-dir", default=CACHE_DIR,
                   help=f"MultiWOZ 2.4 cache directory (default: {CACHE_DIR})")
    return p


if __name__ == "__main__":
    # Print the split statistics without touching the model or the GPU.
    # The dialect changes the text of each prompt but never how many samples
    # or dialogues a node receives, so these numbers hold for every model.
    args = build_parser().parse_args()
    dialect = resolve_model(args.model).dialect
    if args.split_type in ("noniid", "both"):
        print("### Non-IID split ###")
        get_split("noniid", dialect, args.n_device, args.seed, args.cache_dir)
    if args.split_type in ("iid", "both"):
        print("\n### IID split ###")
        get_split("iid", dialect, args.n_device, args.seed, args.cache_dir)
