# Measured results

[Back to the README](../README.md)

---

Results measured with Qwen3-8B on an RTX 5090, evaluated with `--no-fast-inference`. Numbers are only comparable within one machine and software stack.

## Quantitative

Every node's adapter is scored on the same all-domains test set. With no
arguments, the latest checkpoint of every node is evaluated, using the model
recorded in `wafl_meta.json`:

```bash
uv run wafl_llm_dst_eval.py
```

For a model other than the default, point it at that model's directory:

```bash
uv run wafl_llm_dst_eval.py --model ministral3-8b
```

### Results

Qwen3-8B, domain-based Non-IID split, 10 nodes, RWP contact pattern, 300 test
turns:

```
  node 0: JGA=83.00%  SlotF1=97.90%  parse_fail= 0.0%
  node 1: JGA=77.67%  SlotF1=97.04%  parse_fail= 0.0%
  node 2: JGA=76.67%  SlotF1=97.29%  parse_fail= 0.0%
  node 3: JGA=79.33%  SlotF1=97.82%  parse_fail= 0.0%
  node 4: JGA=80.00%  SlotF1=97.78%  parse_fail= 0.0%
  node 5: JGA=78.33%  SlotF1=97.72%  parse_fail= 0.0%
  node 6: JGA=82.33%  SlotF1=98.10%  parse_fail= 0.0%
  node 7: JGA=79.00%  SlotF1=97.92%  parse_fail= 0.0%
  node 8: JGA=81.00%  SlotF1=98.14%  parse_fail= 0.0%
  node 9: JGA=80.33%  SlotF1=98.05%  parse_fail= 0.0%
============================================================
WAFL-LLM DST results  (qwen3-8b, split=noniid, round=300, 10 nodes)
============================================================
Mean JGA    :  79.77%
Min  JGA    :  76.67%
Max  JGA    :  83.00%
Std  JGA    :   1.89%   (smaller means the nodes converged together)
Mean SlotF1 :  97.78%
============================================================
```

### Convergence

`--rounds-curve` over the saved checkpoints:

| round | mean JGA | min | max | std JGA | mean slot F1 |
| ----- | -------- | --- | --- | ------- | ------------ |
| 0 (self-train only) | 53.57 % | 31.67 % | 67.67 % | 13.93 % | 91.01 % |
| 50 | 69.63 % | 31.67 % | 79.00 % | 13.38 % | 95.05 % |
| 100 | 76.30 % | 73.00 % | 80.67 % | 2.50 % | 97.06 % |
| 150 | 80.57 % | 78.67 % | 82.67 % | 1.19 % | 97.92 % |
| 200 | 80.43 % | 75.67 % | 84.33 % | 2.34 % | 97.91 % |
| 250 | 79.67 % | 77.67 % | 81.67 % | 1.31 % | 97.82 % |
| 300 | 79.77 % | 76.67 % | 83.00 % | 1.89 % | 97.78 % |

Collaboration is worth **+26 points of mean JGA** over self-training alone, and
the spread between nodes falls by roughly a factor of seven. Both numbers
matter, and the second is the one specific to WAFL: the nodes do not merely get
better on average, they converge on a shared model rather than staying as ten
local specialists. The mean has flattened by round 150 and the remaining
movement is noise on a 300-turn test set, which is why `--rounds 300` is the
default.

### Where the gain comes from

Per node, comparing round 0 with round 300:

| node | domain | local samples | round 0 | round 300 | gain |
| ---- | ------ | ------------- | ------- | --------- | ---- |
| 0 | restaurant | 9228 | 64.33 % | 83.00 % | +18.7 |
| 1 | restaurant | 9089 | 67.00 % | 77.67 % | +10.7 |
| 2 | hotel | 9177 | 67.67 % | 76.67 % | +9.0 |
| 3 | hotel | 9062 | 57.33 % | 79.33 % | +22.0 |
| 4 | train | 7450 | 67.00 % | 80.00 % | +13.0 |
| 5 | train | 7292 | 62.67 % | 78.33 % | +15.7 |
| 6 | attraction | 1940 | 44.67 % | 82.33 % | +37.7 |
| 7 | attraction | 1934 | 41.00 % | 79.00 % | +38.0 |
| 8 | taxi | 817 | 32.33 % | 81.00 % | **+48.7** |
| 9 | taxi | 789 | 31.67 % | 80.33 % | **+48.7** |

At round 0 a node's accuracy is almost entirely explained by how much data it
happens to hold: the correlation between local sample count and JGA is
**+0.95**. The `taxi` nodes, with under 1000 samples each, sit at barely 32 %.

By round 300 that correlation has fallen to **-0.37**, which on ten points is
indistinguishable from none. The data-poor nodes gained nearly 49 points and
finished level with the data-rich ones; node 8 (817 samples) ends up *ahead* of
node 2 (9177 samples). The advantage of holding more data has been erased by
the exchange, which is precisely the outcome WAFL is aiming for.

Other things you may want:

```bash
# the self-training-only baseline, before any model exchange
uv run wafl_llm_dst_eval.py --round 0

# convergence curve over every saved round, starting from that baseline
uv run wafl_llm_dst_eval.py --rounds-curve

# one specific round, a subset of nodes, on more test turns
uv run wafl_llm_dst_eval.py --round 100 --nodes 0,4,9 --max-test 1000

# the IID baseline
uv run wafl_llm_dst_eval.py --split-type iid
```

Bear in mind the cost: the script decodes greedily, one turn at a time, so the
runtime scales with nodes × turns. `--rounds-curve` multiplies that by the
number of saved rounds, so pair it with a small `--max-test`.

* **JGA (Joint Goal Accuracy)** — the fraction of turns where the predicted
  belief state matches the reference *exactly*. One wrong or missing slot fails
  the whole turn, which makes it a demanding metric and the standard one for
  DST.
* **Slot F1** — micro-averaged precision/recall/F1 over individual
  `slot=value` pairs. It degrades gracefully, so it shows partial progress that
  JGA hides.
* **Std JGA** — the spread across nodes. This is the WAFL-specific number: it
  should shrink as the rounds go on, indicating that the nodes converged on a
  shared model instead of drifting apart into local specialists.
* **parse_fail** — the fraction of generations from which no JSON object could
  be recovered at all. It should be near zero after training; a high value
  means the run is undertrained rather than inaccurate.

---

## Qualitative

`wafl_llm_dst_eval.py` tells you *how good* the network is. It does not tell you
*what a node is getting wrong*, and a JGA number alone cannot distinguish a
model that misses slots from one that invents them — both just look like a lower
score. `wafl_llm_dst_inspect.py` fills that gap: it replays the test dialogues
through a single adapter and prints, turn by turn, the conversation the model
saw, the JSON it produced, the reference JSON, and a slot-level diff.

Because a full report over every node would be unreadable, **the round and the
node are both required**:

```bash
uv run wafl_llm_dst_inspect.py --round 50 --node 3
```

Each turn is rendered like this:

```
--- turn 1 --- MISMATCH
  conversation so far:
    user  | i need a cheap hotel in the north
    state | {"hotel-area": "north", "hotel-pricerange": "cheap"}
  USER  | book it for 3 nights
  slots:
    OK       hotel-area = north
    MISSING  hotel-book_stay = '3'  (not predicted)
    MISSING  hotel-pricerange = 'cheap'  (not predicted)
```

Every slot falls into one of four buckets, and which one dominates is the
diagnosis:

| Outcome | Meaning |
| ------- | ------- |
| `OK`       | slot predicted with the right value |
| `WRONG`    | slot found, value wrong |
| `MISSING`  | slot in the reference, not predicted |
| `SPURIOUS` | slot predicted, not in the reference |

A pile of `MISSING` means the model is under-predicting — very common early on,
and the classic symptom of a Non-IID node being asked about a domain it never
saw. A pile of `SPURIOUS` means it is inventing slots, and `WRONG` means it
locates the slot but not the value. The example above shows the most
characteristic DST failure of all: the model reports only what the *latest*
utterance mentioned and drops the state it had already accumulated, which fails
JGA on every subsequent turn of the dialogue.

Like the quantitative script, 300 test turns are run by default. Only the first
20 are printed, but **all 300 count towards the summary** at the end:

```
====================================================================
Summary over all 300 turns (qwen3-8b, round 50, node 3)
====================================================================
Exact-match turns : ../300 (JGA ..%)
Slot precision    : ..%
Slot recall       : ..%
Slot F1           : ..%
Parse failures    : .. (..%)
Slot outcomes     : .. ok, .. wrong value, .. missing, .. spurious
====================================================================
```

Useful variations:

```bash
# only the turns that failed, which is usually what you want
uv run wafl_llm_dst_inspect.py --round 50 --node 3 --only-errors

# compare a node before and after collaboration: round 0 is self-train only
uv run wafl_llm_dst_inspect.py --round 0   --node 9 --only-errors
uv run wafl_llm_dst_inspect.py --round 150 --node 9 --only-errors

# a different model's run
uv run wafl_llm_dst_inspect.py --model llama31-8b --round 50 --node 3

# chase parse failures by showing the raw generations
uv run wafl_llm_dst_inspect.py --round 50 --node 3 --max-test 30 --show-raw

# save a shareable report, or per-turn records for your own analysis
uv run wafl_llm_dst_inspect.py --round 50 --node 3 --out-md report.md
uv run wafl_llm_dst_inspect.py --round 50 --node 3 --out-jsonl turns.jsonl
```

The `--round 0` versus `--round 150` pair on a data-poor node such as node 8 or
node 9 (the `taxi` nodes, with under 1000 samples each) is the most direct way
to *see* what WAFL bought you: at round 0 that node has only ever seen taxi
dialogues, so hotel and restaurant slots come back `MISSING`; if collaboration
worked, those same slots are filled in afterwards.

### What a Non-IID node actually gets wrong

Node 9 is one of the two `taxi` nodes, with 789 local samples, and at round 0
it has never trained on anything but taxi dialogues. Running

```bash
uv run wafl_llm_dst_inspect.py --round 0 --node 9 --only-errors --max-test 50
```

gives:

```
Exact-match turns : 11/50 (JGA 22.00%)
Slot precision    : 89.14%
Slot recall       : 76.28%
Slot F1           : 82.21%
Parse failures    : 4 (8.0%)
Slot outcomes     : 312 ok, 2 wrong value, 95 missing, 36 spurious
```

Three things stand out, and none of them is visible in the JGA number alone.

**It answers about taxis no matter what it was asked.** Four turns fail to
parse, and they all look like this:

```
--- turn 0 --- MISMATCH  [PARSE FAILED]
  USER  | I'm looking for a place to stay. It needs to be a guesthouse and include free wifi.
  slots:
    MISSING  hotel-internet = 'yes'  (not predicted)
    MISSING  hotel-type = 'guest house'  (not predicted)
  raw output: {"taxi-destination": "london", "taxi-destination": "london", "taxi-destination":
               "london", "taxi-destination": "london", ...
```

Asked about a guesthouse, the node emits taxi slots until it runs out of
tokens. This is what overfitting to one Non-IID slice looks like from the
inside.

**Most errors are schema errors, not comprehension errors.** The model
understands the conversation and extracts the right values; it just does not
know what the slots for other domains are called, so it invents plausible
names:

| what the node emitted | the actual MultiWOZ slot |
| --------------------- | ------------------------ |
| `hotel-fee` | `hotel-pricerange` |
| `hotel-people`, `hotel-stay` | `hotel-book_people`, `hotel-book_stay` |
| `restaurant-cuisine` | `restaurant-food` |
| `hotel-star-rating` | `hotel-stars` |
| `hotel-amenities: "free parking,wifi"` | `hotel-internet` + `hotel-parking` |
| `restaurant-price: "high"` | `restaurant-pricerange: "expensive"` |
| `day`, `time`, `partySize` (no domain prefix) | `restaurant-book_day/_time/_people` |

Turn 9 is the clearest case: the node correctly picks "4 people", "15:00" and
"wednesday" out of the utterance, then files them under `partySize`, `time` and
`day`, scoring three `MISSING` and three `SPURIOUS` for information it
extracted perfectly. A large share of the 95 missing and 36 spurious slots are
pairs like this — the same value under the wrong key. What node 9 lacks is not
reasoning ability but the **slot vocabulary of the domains it never saw**, and
that is exactly the thing the other nodes have and can pass over.

**Values are already normalized correctly.** Only 2 of 445 slot decisions are
`WRONG`: `guesthouse` for `guest house`, and `free` for `yes`. The base model's
own competence is intact; what fine-tuning has to supply is the task-specific
convention.

Note that 22.00 % here is below the 31.67 % node 9 scores over the full test
set, because the first 50 test turns happen to be hotel- and
restaurant-heavy — the worst possible sample for a taxi specialist.

### The same node after collaboration

Round 150 is where mean JGA peaks, and round 300 is the default stopping point.
Same node, same 50 turns, all three checkpoints:

```bash
uv run wafl_llm_dst_inspect.py --round 150 --node 9 --only-errors --max-test 50
```

| | round 0 | round 150 | round 300 |
| --- | --- | --- | --- |
| JGA over these 50 turns | 22.00 % | **72.00 %** | 68.00 % |
| slot precision | 89.14 % | 98.00 % | **98.50 %** |
| slot recall | 76.28 % | 96.09 % | 96.09 % |
| slot F1 | 82.21 % | 97.04 % | **97.28 %** |
| parse failures | 4 (8.0 %) | **0** | **0** |
| `OK` slots | 312 | 393 | 393 |
| `MISSING` | 95 | 9 | 11 |
| `SPURIOUS` | 36 | **1** | **1** |
| `WRONG` | 2 | 7 | 5 |

**Invented slot names are gone.** `SPURIOUS` drops from 36 to 1, and the single
survivor is not a schema error at all. Not one `hotel-fee`, `partySize` or
`restaurant-cuisine` remains. The slot vocabulary of the four domains node 9
never trained on arrived through the exchange, which is the mechanism the
convergence curve only shows in aggregate.

**The taxi fixation is gone.** Zero parse failures: the node no longer answers
hotel questions with a wall of `taxi-destination`.

**The error profile has inverted.** At round 0 the node under-predicted
massively (95 missing) and mislabelled what it did find (36 spurious), while
almost never getting a value wrong (2). After collaboration, missing and
spurious have collapsed. Errors have moved from "does not know the schema" to
"knows the schema, disputes the value" — a much later-stage kind of mistake.

**Round 150 versus round 300 is noise, not decay.** JGA falls by 4 points while
every slot-level metric holds or improves, which looks contradictory until you
count the errors: both checkpoints make exactly **17** slot mistakes, just
distributed differently (7 wrong + 9 missing + 1 spurious at 150, versus 5 + 11
+ 1 at 300). What changed is that the model stopped guessing. On turn 13 the
round-150 adapter fills `hotel-name` with `alexander bed and breakfast`, an
invented guess; the round-300 adapter leaves it blank. That converts two
`WRONG` into two `MISSING`, lifting precision and leaving recall untouched.
Because JGA is exact-match over the whole turn, two turns flipping is worth 4
points — which is a good reminder to read JGA and slot F1 together, and why the
plateau between rounds 150 and 300 in the convergence curve should be read as
converged rather than as slowly degrading.

**What is left.** Of the 16 remaining mismatched turns at round 300, six hinge
on **entity names the model has no way to know**:

```
--- turn 18 --- MISMATCH
  USER  | Sure! How close is the Cafe from my current location?
  slots:
    OK       restaurant-area = centre
    OK       restaurant-food = italian
    MISSING  restaurant-name = 'clowns cafe'  (not predicted)
    OK       restaurant-pricerange = expensive
```

The user says "the Cafe"; the reference expects `clowns cafe`, a name that only
ever appeared in the *system's* reply after a database lookup. This project's
prompts deliberately put belief states rather than system utterances in the
history (see `dialogue_to_samples()` in `mwz24_data.py`), so the venue name is
simply not in the model's context. The same cause explains the missing
`hotel-name` on turns 13, 31 and 47 and the wrong `taxi-departure` on turn 26.
This is a ceiling imposed by the prompt design, not by WAFL, and feeding the
system replies back into the history is the obvious thing to try next.

Four more turns are `dontcare` judgements — "It doesn't need to have free
parking" is annotated `dontcare` rather than `no` — and the rest are surface
conventions: `guesthouse` versus `guest house` (twice) and `the cow pizza
kitchen and bar` versus `cow pizza kitchen and bar`.

---
