"""
mwz24_data.py
Download and preprocessing utilities for MultiWOZ 2.4.

This module is shared by the training script (wafl_llm_dst_train.py), the data
splitter (wafl_llm_data_split.py), the evaluation script
(wafl_llm_dst_eval.py) and the inspection script (wafl_llm_dst_inspect.py), so
that all of them build exactly the same prompts.

Prompt construction itself lives in model_registry.py, because it depends on
which model family is in use. Everything here is model-independent apart from
the `dialect` argument that gets passed through.

MultiWOZ 2.4 is not distributed on the Hugging Face Hub; it is published on
GitHub. Its file format is identical to MultiWOZ 2.1:

    data.json         : {dialogue_id: {"goal": ..., "log": [turn, ...]}}
    valListFile.json  : dialogue IDs used for validation (one per line)
    testListFile.json : dialogue IDs used for testing (one per line)

Inside "log", user and system turns alternate:

    log[0] = user, log[1] = system, log[2] = user, ...

The "metadata" field of a system turn holds the belief state accumulated up to
the preceding user turn, which is what we use as the DST label.
"""

import io
import json
import os
import urllib.request
import zipfile

from model_registry import (
    MAX_HISTORY_TURNS,
    answer_suffix,
    build_prompt_text,
)

MWZ24_URL = "https://github.com/smartyfh/MultiWOZ2.4/archive/refs/heads/main.zip"
CACHE_DIR = "./mwz24_data"

# Placeholder strings that mean "this slot has no value" in the belief state.
NONE_VALUES = {"", "not mentioned", "none"}


# ===========================
# Download and extraction
# ===========================

def download_mwz24(cache_dir: str = CACHE_DIR) -> str:
    """
    Download and extract MultiWOZ 2.4 from GitHub.

    Extraction is skipped if the dataset is already present.
    Returns the path of the directory that holds data.json.
    """
    data_dir = os.path.join(cache_dir, "MULTIWOZ2.4")

    if os.path.exists(os.path.join(data_dir, "data.json")):
        print(f"Using cached MultiWOZ 2.4 at {data_dir}")
        return data_dir

    os.makedirs(cache_dir, exist_ok=True)

    print("Downloading MultiWOZ 2.4 from GitHub ...")
    repo_zip_path = os.path.join(cache_dir, "repo.zip")
    urllib.request.urlretrieve(MWZ24_URL, repo_zip_path)

    print("Extracting ...")
    # The repository archive contains another archive, data/MULTIWOZ2.4.zip,
    # which is the one holding the actual dataset files.
    with zipfile.ZipFile(repo_zip_path) as repo_zip:
        inner_name = "MultiWOZ2.4-main/data/MULTIWOZ2.4.zip"
        with repo_zip.open(inner_name) as inner_file:
            inner_bytes = inner_file.read()

    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as data_zip:
        data_zip.extractall(cache_dir)

    os.remove(repo_zip_path)
    print(f"Done. Data at {data_dir}")
    return data_dir


# ===========================
# Belief state extraction
# ===========================

def extract_belief_state(metadata: dict) -> dict:
    """
    Turn the "metadata" field of a system turn into a flat belief-state dict.

    The metadata is structured as:
      {
        "hotel": {
          "book": {"booked": [...], "stay": "3", "day": "tuesday", ...},
          "semi": {"pricerange": "cheap", "parking": "yes", ...}
        },
        "restaurant": {...},
        ...
      }

    Example of the returned value:
      {"hotel-pricerange": "cheap", "hotel-book_stay": "3", ...}
    """
    belief = {}
    for domain, dv in metadata.items():
        # "semi" holds the search-constraint slots.
        for slot, value in dv.get("semi", {}).items():
            if value and value not in NONE_VALUES:
                belief[f"{domain}-{slot}"] = value
        # "book" holds the reservation slots. "booked" is the booking result
        # rather than a user constraint, so it is dropped.
        for slot, value in dv.get("book", {}).items():
            if slot == "booked":
                continue
            if value and value not in NONE_VALUES:
                belief[f"{domain}-book_{slot}"] = value
    return belief


# ===========================
# Sample generation
# ===========================

def dialogue_to_samples(dialogue: dict, dialect: str) -> list:
    """
    Expand one dialogue into a list of DST samples, one per user turn.

    Each sample looks like:
      {
        "prompt": prompt string, up to the start of the assistant answer,
        "label":  belief state as a JSON string (the training target),
        "gold":   belief state as a dict (used by the metrics),
        "utterance": the user utterance this sample is about,
      }

    `dialect` selects the chat format; see model_registry.DIALECTS.
    """
    samples = []
    log = dialogue.get("log", [])
    history = []

    # Turns alternate: user at even indices, system at odd indices.
    # The belief state for user turn i is stored in system turn i+1.
    for i in range(0, len(log) - 1, 2):
        user_turn = log[i]
        system_turn = log[i + 1]

        user_text = user_turn.get("text", "").strip()
        belief_state = extract_belief_state(system_turn.get("metadata", {}))
        label = json.dumps(belief_state, ensure_ascii=False, sort_keys=True)

        prompt = build_prompt_text(dialect, history, user_text,
                                   MAX_HISTORY_TURNS)

        samples.append({
            "prompt":    prompt,
            "label":     label,
            "gold":      belief_state,
            "utterance": user_text,
        })

        # Grow the history. The assistant side of the history is the belief
        # state rather than the system utterance, so that the model is
        # conditioned on the state it predicted so far. Swap in
        # system_turn["text"] here if you would rather condition on the
        # natural-language system replies.
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": label})

    return samples


def to_text(sample: dict, dialect: str) -> str:
    """Join a (prompt, label) pair into a single training string."""
    return sample["prompt"] + sample["label"] + answer_suffix(dialect)


def load_split(split: str, dialect: str, max_samples: int = None,
               cache_dir: str = CACHE_DIR) -> list:
    """
    Return the list of samples for one MultiWOZ 2.4 split.
    split: "train" / "validation" / "test"
    """
    data_dir = download_mwz24(cache_dir)

    with open(os.path.join(data_dir, "data.json"), encoding="utf-8") as f:
        data = json.load(f)

    with open(os.path.join(data_dir, "valListFile.json")) as f:
        val_ids = {line.strip() for line in f if line.strip()}
    with open(os.path.join(data_dir, "testListFile.json")) as f:
        test_ids = {line.strip() for line in f if line.strip()}

    if split == "validation":
        target_ids = val_ids
    elif split == "test":
        target_ids = test_ids
    elif split == "train":
        target_ids = set(data.keys()) - val_ids - test_ids
    else:
        raise ValueError(f"Unknown split: {split}")

    samples = []
    for dialogue_id in sorted(target_ids):
        if dialogue_id not in data:
            continue
        samples.extend(dialogue_to_samples(data[dialogue_id], dialect))
        if max_samples and len(samples) >= max_samples:
            break

    if max_samples:
        samples = samples[:max_samples]

    print(f"MultiWOZ 2.4 [{split}]: {len(samples)} samples")
    return samples


if __name__ == "__main__":
    # Sanity check of the preprocessing, and a look at every prompt dialect.
    import argparse

    from model_registry import DIALECTS, MODEL_PRESETS, resolve_model

    parser = argparse.ArgumentParser(
        description="Preview MultiWOZ 2.4 samples and the prompt format.")
    parser.add_argument("--model", default=None,
                        help="preset alias or Hugging Face id; selects the "
                             "dialect (default: show every dialect)")
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    args = parser.parse_args()

    dialects = ([resolve_model(args.model).dialect] if args.model
                else sorted(DIALECTS))

    for dialect in dialects:
        users = [a for a, s in MODEL_PRESETS.items() if s.dialect == dialect]
        print("\n" + "=" * 70)
        print(f"dialect: {dialect}  ({DIALECTS[dialect]['description']})")
        print(f"used by: {', '.join(users) if users else '-'}")
        print("=" * 70)
        samples = load_split("validation", dialect, max_samples=2,
                             cache_dir=args.cache_dir)
        s = samples[-1]
        print("--- prompt (tail) ---")
        print(s["prompt"][-320:])
        print("--- training text (tail) ---")
        print(to_text(s, dialect)[-320:])
