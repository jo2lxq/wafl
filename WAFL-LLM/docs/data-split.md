# Data split

[Back to the README](../README.md)

---

How MultiWOZ 2.4 is turned into per-node datasets, and what the resulting distribution looks like.

The dataset is **MultiWOZ 2.4**, a multi-domain task-oriented dialogue corpus of
about 10k human-human dialogues. It is downloaded automatically on first run
from <https://github.com/smartyfh/MultiWOZ2.4> into `./mwz24_data`. The
official `valListFile.json` / `testListFile.json` define the validation and test
sets; everything else is training data.

Every **user turn** becomes one training sample: the prompt holds the recent
dialogue history plus the current utterance, and the target is the belief state
as a JSON object, e.g.

```json
{"hotel-area": "centre", "hotel-book_people": "2", "hotel-pricerange": "cheap"}
```

### Splitting over the nodes

Each dialogue is first assigned a **primary domain**: the domain that appears
most often across the belief states of its turns. Then:

* **`noniid` (default) — domain-based.** The five dominant domains
  (`restaurant`, `hotel`, `train`, `attraction`, `taxi`) are handed out to the
  nodes, two nodes per domain at the default `--n-device 10`. A dialogue goes to
  one of the nodes responsible for its primary domain. Dialogues whose primary
  domain is something else (`misc`, `hospital`, `bus`, ...) are spread thinly
  across all nodes so that no data is thrown away.

  This is the interesting case, and it is meant to model the ubiquitous setting
  directly: a device accumulates the kind of dialogue its owner actually has.
  A node biased towards `taxi` sees almost no hotel slots at all, so on its own
  it cannot learn them.

* **`iid` — random.** All dialogues are shuffled and dealt out evenly, so every
  node sees roughly the same domain mixture. This is the baseline that tells you
  how much of the final score is attributable to the collaboration rather than
  to the fine-tuning itself.

Note that the split is **by dialogue, not by turn**, so a dialogue's turns never
end up spread across nodes, and the history in a prompt is always consistent.

### The test set is IID

All nodes are evaluated on the **same** test set, which mixes every domain.
That is the whole measurement: a node that only ever trained on taxi dialogues
is asked about hotels and restaurants too. Success looks like a high mean score
*and* a small standard deviation across nodes — high mean alone could just be
one lucky node, and the spread is what shows the network converged rather than
fragmenting into specialists.

You can inspect the split, including the per-node domain distribution, without
loading a model or touching the GPU:

```bash
uv run wafl_llm_data_split.py --split-type noniid --n-device 10
```

```
===== Data Split: Non-IID (domain-based) =====
  node 0:  9228 samples (1428 dialogues) | top: restaurant:1373, misc:42, hospital:12
  node 1:  9089 samples (1425 dialogues) | top: restaurant:1373, misc:41, hospital:11
  node 2:  9177 samples (1201 dialogues) | top: hotel:1149, misc:41, hospital:11
  node 3:  9062 samples (1200 dialogues) | top: hotel:1148, misc:41, hospital:11
  node 4:  7450 samples (1035 dialogues) | top: train:983, misc:41, hospital:11
  node 5:  7292 samples (1034 dialogues) | top: train:982, misc:41, hospital:11
  node 6:  1940 samples (339 dialogues)  | top: attraction:287, misc:41, hospital:11
  node 7:  1934 samples (338 dialogues)  | top: attraction:286, misc:41, hospital:11
  node 8:   817 samples (219 dialogues)  | top: taxi:167, misc:41, hospital:11
  node 9:   789 samples (219 dialogues)  | top: taxi:167, misc:41, hospital:11
  total: 56778 samples
```

The two nodes assigned to the same domain end up with near-identical sample
counts (the random shuffle just alternates dialogues between them), and the
domains themselves are far from balanced: `restaurant` and `hotel` nodes carry
roughly 9k samples each, while `taxi` nodes get under 1k. That imbalance is
part of the Non-IID challenge on purpose — it mirrors MultiWOZ's real domain
frequencies rather than an artificially balanced split, and it is exactly what
makes the `taxi` nodes' final accuracy on the full test set the interesting
number to look at.

These counts are the same for every model: the prompt dialect changes the text
of each sample but never how many samples a node receives.

---
