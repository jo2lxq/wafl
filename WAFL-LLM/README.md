# WAFL-LLM

Wireless Ad Hoc Federated Learning (WAFL) of an **LLM LoRA adapter** for
**Dialogue State Tracking (DST)** on MultiWOZ 2.4.

Each node holds its own, differently distributed slice of dialogue data, trains
a LoRA adapter on it, and — whenever the mobility pattern brings it within radio
range of another node — exchanges and aggregates **only the LoRA parameters**
with the nodes it met. No parameter server, no global model, no coordinator.

This project is the WAFL-MLP algorithm carried over from a small MLP on MNIST to
an instruction-tuned LLM on a structured language task. Three model families in
two size classes are supported out of the box, so the same experiment can be run
across models by changing one flag.

---

## Background

WAFL is a way of tuning a model **for a local community**, using nothing but the
device-to-device contacts that the community's own movement produces. That is a
convenient property: nobody has to stand up a training service, register the
devices with it, keep it running, or agree on who operates it. A group of
devices that happen to be in the same place — a campus, a hospital, a museum, a
shopping street, a factory floor — can improve their models simply by meeting
each other.

Dialogue State Tracking is a natural flagship LLM application for this idea.

A task-oriented dialogue assistant only works if it can convert what a person
just said into a structured state: which domain they are talking about, which
slots they have filled, and with what values. What that state space looks like
in practice is intensely local. The venues people ask for, the phrasing they
use, the services that exist, the way a booking is described, and the slots that
actually matter differ from one community to the next. A model tuned once,
centrally, on generic data is generic everywhere; a model tuned to a community
is useful in that community.

At the same time, no single device in a community ever sees the whole picture.
One device accumulates restaurant bookings, another mostly train connections,
another taxi rides. Tuned alone, each of them becomes a narrow specialist and
gets worse — not better — at everything else, which is exactly the Non-IID
problem WAFL was designed for. Collaborative tuning is what lets every device
end up with a model that covers the community's whole domain and slot space,
not just the part it happened to observe.

DST is also a good testbed, for concrete reasons:

* **The output is structured and objectively scorable.** Joint Goal Accuracy is
  exact-match on a JSON object, so there is no room for judging quality by feel.
* **The data partitions naturally.** MultiWOZ dialogues carry an inherent domain
  label, which gives a realistic Non-IID split rather than an artificial one.
* **It is small enough to actually run.** With LoRA on a 4-bit base model, ten
  simulated nodes fit on a single consumer GPU.

---

## Supported models

```bash
uv run wafl_llm_dst_train.py --list-models
```

| Alias | Size | Hugging Face id | Prompt dialect | License |
| ----- | ---- | --------------- | -------------- | ------- |
| `qwen3-8b` *(default)* | 8B | `unsloth/Qwen3-8B` | `chatml_think` | Apache-2.0 |
| `qwen3-4b` | 4B | `unsloth/Qwen3-4B-Instruct-2507` | `chatml_plain` | Apache-2.0 |
| `ministral3-8b` | 8B | `unsloth/Ministral-3-8B-Instruct-2512` | `mistral` | Apache-2.0 |
| `ministral3-3b` | 3B | `unsloth/Ministral-3-3B-Instruct-2512` | `mistral` | Apache-2.0 |
| `llama31-8b` | 8B | `unsloth/Meta-Llama-3.1-8B-Instruct` | `llama3` | Llama 3.1 Community |
| `llama32-3b` | 3B | `unsloth/Llama-3.2-3B-Instruct` | `llama3` | Llama 3.2 Community |

Three families across two size classes, which lets you separate "does WAFL work
regardless of model family" from "how does model capacity affect convergence".
Any other Hugging Face id works too — the dialect is guessed from the name, or
set explicitly with `--dialect`.

**The LoRA payload is nearly identical across all six.** At rank 16 on the seven
attention and MLP projections, every 8B-class model here lands around 40 M
adapter parameters, under 100 MB in fp16. Since that payload *is* the
communication cost in WAFL, model families can be compared without the
comparison being confounded by how much data each one has to transmit. The
training script prints the exact figure at start-up.

### Prompt dialects

Each family formats a conversation differently, so prompt construction is
factored into `model_registry.py`:

| Dialect | Format | Used by |
| ------- | ------ | ------- |
| `chatml_think` | ChatML plus an empty `<think>` block | hybrid-reasoning Qwen3 |
| `chatml_plain` | ChatML, no `<think>` block | Qwen3 `*-Instruct-2507`, Qwen2.5 |
| `llama3` | `<\|start_header_id\|>` headers | Llama 3.x |
| `mistral` | `[INST]` with `[SYSTEM_PROMPT]` | Mistral 3 / Ministral 3 |
| `mistral_nosys` | `[INST]`, system folded into the first user turn | Mistral 7B v0.1–v0.3 |

Two details are worth knowing about:

* **Qwen3 is not one dialect.** The hybrid `Qwen3-8B` expects a `<think>` block
  and is put into non-thinking mode by pre-filling an empty one. The
  `Qwen3-4B-Instruct-2507` refresh never emits `<think>` at all, so inserting
  the block there would be wrong. Hence two ChatML dialects rather than one.
* **Older Mistral has no system role.** Mistral 7B v0.1–v0.3 reject a system
  message outright, so `mistral_nosys` folds the DST instruction into the first
  user turn. Mistral 3 / Ministral 3 support `[SYSTEM_PROMPT]` properly and use
  the plain `mistral` dialect.

The templates are written out by hand rather than going through
`apply_chat_template`, so that training and generation use byte-identical
strings and the data pipeline needs no tokenizer. The risk of that approach is
silently drifting from what the model was post-trained on, so at start-up the
training script **checks its template against the tokenizer's own** and warns on
any mismatch:

```
  template check: OK (dialect 'chatml_think' matches the tokenizer's chat template)
```

If you see `MISMATCH`, the check prints the exact character where the two
strings first diverge, with context on both sides, so you can fix
`DIALECTS[...]` in `model_registry.py`.

**One expected, benign difference.** Llama 3.x chat templates inject two lines
ahead of the system message:

```
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a Dialogue State Tracking (DST) model. ...
```

The `llama3` dialect here leaves those out on purpose, because Llama 3.2 fills
`Today Date` with the *current* date — which would make prompts depend on the
day a run happened to start, and would make a Llama run differ from the other
families in prompt content rather than just markup. Since training and
evaluation both use the same fixed prefix, LoRA absorbs it and results stay
valid. The checker recognises this specific case and reports it without warning:

```
  template check: OK for 'llama3', apart from the Llama 'Cutting Knowledge Date' /
    'Today Date' preamble, which is omitted on purpose ...
```

---

## How it works

### One base model, ten adapters

The base model is **frozen and identical on every node**, so it is never
exchanged. Only a LoRA adapter is trainable, and the adapter is what travels
over the ad hoc link — on the order of 10⁷ parameters instead of 10⁹. This is
what makes the idea plausible on a real device-to-device link, and it is the
main difference from WAFL-MLP, which exchanges the whole model.

Because a simulation of ten nodes does not need ten copies of an 8B model, the
script keeps **one** model in memory plus ten dictionaries of LoRA parameters,
and swaps the relevant dictionary in whenever a node's turn comes up. Each node
also keeps its own AdamW optimizer state across rounds.

### Phase 1: pre-self-training

Before any exchange happens, every node trains briefly on its own local data.
The WAFL paper reports that this initial self-training makes the subsequent
aggregation phase climb much faster. It is deliberately short, and it is never
repeated later — prolonged solitary training just overfits a node back onto its
own Non-IID slice.

The state at the end of this phase is checkpointed as **round 0**, giving you
the self-training-only baseline to compare every later round against.

### Phase 2: WAFL rounds

Each round reads one entry of the contact pattern, which says who is within
radio range of whom at that moment, and then:

1. **Exchange and aggregate.** Every node with at least one neighbour pulls in
   its neighbours' adapters and mixes them into its own, using the WAFL
   aggregation rule:

   ```
   theta'(n) = theta(n) + lambda * sum_{k in nbr(n)} ( theta(k) - theta(n) ) / ( |nbr(n)| + 1 )
   ```

   With one neighbour and `lambda = 1.0` this lands exactly halfway between the
   two adapters; with several neighbours it approaches their mean. All nodes are
   updated from the same pre-round snapshot, so the round is simultaneous rather
   than sequential.

2. **Adjust locally.** A node that met somebody then trains the aggregated
   adapter on its own data for a fraction of an epoch. This is the step that
   turns a blend of other people's adapters into something that also still fits
   the local data — and, over many rounds, pulls every node towards a model that
   minimizes the loss over the *virtually merged* dataset of the whole network.

3. **Nodes that met nobody do nothing.** No aggregation and, importantly, no
   local training either, for the overfitting reason above.

Checkpoints of all nodes are written every `--save-every` rounds, which is what
the evaluation script reads.

### Contact patterns

A contact pattern is a JSON array with one entry per round. Each entry maps a
node ID (as a string) to the list of nodes in radio range during that round:

```json
[
  {"0": [7], "1": [], "2": [5, 9], "3": [], "...": []},
  {"0": [],  "1": [4], "2": [5],    "3": [8], "...": []}
]
```

The default is `contact_pattern/rwp_n10_a0500_r100_p10_s01.json`, a random
waypoint mobility trace. The filename encodes the simulation parameters:

| Part    | Meaning |
| ------- | ------- |
| `rwp`   | random waypoint mobility |
| `n10`   | 10 nodes |
| `a0500` | 500 m × 500 m movement area |
| `r100`  | 100 m radio range |
| `p10`   | 10 epochs of pause time at each waypoint |
| `s01`   | random seed 1 |

RWP is a good default because it lets any node eventually meet any other node,
which matches people moving around a shopping mall or a campus. Static
topologies (`static_line`, `static_ringstar`, ...) and community-structured
traces (`cse*`) can be dropped in with `--contact-file` instead; the static ones
converge fastest, sparser mobility slowest.

Note that the number of rounds actually run is
`min(--rounds, len(contact_pattern))`. The default file has 10000 entries, well
beyond the default `--rounds 300`, so raising `--rounds` later needs no change
to the contact file — just a longer run.

---

## Data split

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

## Requirements

* NVIDIA GPU with CUDA. An 8B model in 4-bit with LoRA, gradient checkpointing
  and a sequence length of 2048 is comfortable on a 24 GB card; the 3B/4B
  presets need considerably less. Smaller cards may want `--max-seq-len 1024`
  or `--batch-size 1`.
* Python 3.12
* About 20 GB of disk per model for weights, dataset and per-round checkpoints
  (10 nodes × ~90 MB per saved round adds up — tune `--save-every`).

A 300-round run trains every scheduled node in every round and takes a while on
a single GPU. 300 is the default because that is comfortably past convergence
for the default RWP pattern: the measured curve below flattens by round 150 and
the spread between nodes has already collapsed by round 100. Static topologies
converge faster still. For a first pass, cut it down with `--rounds`,
`--local-train-epochs` and `--max-samples-per-node`, or start with a 4B-class
preset.

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/).

```bash
# in the WAFL-LLM directory
uv init
uv python pin 3.12
```

`uv init` will create a `pyproject.toml`; if it also creates a `main.py`, you
can delete it.

### Pin torch and torchvision to your CUDA driver

`uv add torch` alone lets uv pick whatever CUDA build is newest, which can be
newer than what your GPU driver actually supports — that fails at import time
(`CUDA initialization: The NVIDIA driver on your system is too old`) or shows
up as an obscure `undefined symbol` error from `libtorch_cuda.so`. To avoid
this, pin `torch` to a build that matches your driver's CUDA version.

First check what your driver supports:

```bash
nvidia-smi
```

Look at `CUDA Version: XX.X` in the top-right corner. Then add the matching
[PyTorch index](https://docs.astral.sh/uv/guides/integration/pytorch/) to
`pyproject.toml` — for CUDA 12.8 drivers:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux'" },
]
torchvision = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux'" },
]
```

(swap `cu128` for `cu118`, `cu126`, or `cu130` if that is what `nvidia-smi`
reports instead).

**Add both `torch` and `torchvision` explicitly**, one at a time, and check
after each:

```bash
uv add torch
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

uv add torchvision
uv run python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

Both version strings must show the same suffix, e.g. `2.11.0+cu128` and
`0.26.0+cu128`. This step matters even though nothing in this project imports
`torchvision` directly — `transformers` pulls it in regardless, and
`[tool.uv.sources]` in uv only overrides packages that are **direct**
dependencies of the project. If `torchvision` is left to be resolved purely as
someone else's transitive dependency, it silently comes from the default index
instead of the pinned CUDA one, the two builds disagree, and you get
`RuntimeError: operator torchvision::nms does not exist` the first time
`unsloth`/`transformers` imports it. Adding it explicitly, as above, is what
fixes that.

### The rest of the dependencies

```bash
uv add unsloth
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

uv add datasets transformers trl peft accelerate bitsandbytes tqdm
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Adding packages one group at a time and re-checking `torch.cuda.is_available()`
after each is worth the extra few seconds: if a later package quietly drags in
an incompatible `torch` or `nvidia-nccl-cu12`/`nvidia-nccl-cu13` build, you find
out immediately instead of after a long download and a confusing traceback.
`uv pip list | grep -Ei "torch|nccl|nvidia"` is a quick way to check that
nothing has ended up with mismatched CUDA-version suffixes.

Finally, check that the data pipeline works, which will also download
MultiWOZ 2.4 and print a sample prompt in every dialect:

```bash
uv run mwz24_data.py
```

To see just one model's format:

```bash
uv run mwz24_data.py --model ministral3-8b
```

### Contact pattern

The contact pattern files are the same ones used throughout this repository.
Copy the one you want next to the scripts:

```bash
mkdir -p contact_pattern
cp ../WAFL-MLP/data/contact_pattern/rwp_n10_a0500_r100_p10_s01.json contact_pattern/
```

Any other trace works too, as long as its node count matches `--n-device`.

### Troubleshooting

**Generation crashes with a broadcast shape error.** If evaluation dies inside
unsloth's attention kernel like this:

```
File ".../unsloth/models/qwen3.py", line 313, in Qwen3Attention_fast_forward_inference
    Qn *= cos
RuntimeError: output with shape [1, 32, 1, 128] doesn't match the broadcast
shape [1, 32, 87, 128]
```

then unsloth is older than the installed `transformers`. The `1` in the first
shape is a single decode step and the `87` is the prompt length: unsloth has
routed the whole prompt through its decode-only kernel. It picks between its
prefill and decode kernels by testing whether a KV cache was handed in, and
newer transformers passes an empty cache object during prefill instead of
`None`, which older unsloth builds read as "already decoding".

Upgrade both unsloth packages, which are versioned as a pair:

```bash
uv add -U unsloth unsloth_zoo
uv run python -c "import unsloth; print(unsloth.__version__)"
```

LoRA weights do not depend on the library version, so **existing checkpoints
stay valid** and there is no need to retrain.

If the upgrade is not possible — newer GPUs can pin you to a CUDA build that
only an older unsloth supports — generate with the cache disabled instead:

```bash
uv run wafl_llm_dst_eval.py --no-fast-inference
```

This passes `use_cache=False`, which keeps `past_key_values` at `None` and so
stays off the broken path. It is slower, since nothing is cached between steps,
but the results are identical. The same flag works on
`wafl_llm_dst_inspect.py`.

**Only some models are affected.** unsloth ships hand-written inference kernels
for Qwen and Llama, so those hit the bug; Ministral 3 loads through `FastModel`
and falls back to stock transformers, which is why it evaluates cleanly on the
same broken install. If you need the fallback, use it for every model so the
comparison stays like-for-like.

**`Unsloth should be imported before [transformers, peft]`.** unsloth patches
classes inside transformers and peft as it is imported, so importing it late
costs speed and GPU memory, though not correctness. Every entry point already
does `import unsloth` above its other imports, so this should not appear. If it
does, check that nothing was reordered above that line in the script you are
running.

**Training and evaluation may use different unsloth versions** without any
problem, since only adapter tensors are stored. It is still worth keeping them
aligned so that tokenization and prompt handling stay identical.

---

## Training

Defaults are Qwen3-8B, the domain-based Non-IID split, 10 nodes, the RWP contact
pattern above, and 300 rounds:

```bash
uv run wafl_llm_dst_train.py
```

Switching models is one flag:

```bash
uv run wafl_llm_dst_train.py --model qwen3-4b
uv run wafl_llm_dst_train.py --model ministral3-8b
uv run wafl_llm_dst_train.py --model llama31-8b
uv run wafl_llm_dst_train.py --model llama32-3b
```

Each writes to its own directory (`./wafl-<model>-dst-<split-type>`), so runs
never overwrite each other and several models can be compared side by side.

Some useful variations:

```bash
# short smoke test before committing to a long run
uv run wafl_llm_dst_train.py --rounds 5 --max-samples-per-node 200 --save-every 5

# the IID baseline, for comparison
uv run wafl_llm_dst_train.py --split-type iid

# a different mobility trace and a gentler aggregation coefficient
uv run wafl_llm_dst_train.py \
    --contact-file ./contact_pattern/static_ringstar_n10.json \
    --fl-coefficiency 0.1

# any Hugging Face model that is not in the preset list
uv run wafl_llm_dst_train.py --model unsloth/Qwen3-14B
uv run wafl_llm_dst_train.py --model some-org/custom-model --dialect llama3

# smaller GPU
uv run wafl_llm_dst_train.py --batch-size 1 --grad-acc 4 --max-seq-len 1024
```

### Options

```
model:
  --model MODEL              preset alias or any Hugging Face id
                             (default: qwen3-8b); see --list-models
  --list-models              print the built-in model presets and exit
  --dialect DIALECT          override the prompt format; only needed for
                             models outside the preset list whose name is
                             ambiguous. One of: chatml_plain, chatml_think,
                             llama3, mistral, mistral_nosys
  --loader {auto,language,multimodal}
                             which unsloth loader to use; 'auto' follows the
                             preset (default: auto)
  --max-seq-len MAX_SEQ_LEN  maximum sequence length in tokens (default: 2048)
  --lora-rank LORA_RANK      LoRA rank r (default: 16)
  --lora-alpha LORA_ALPHA    LoRA scaling factor alpha (default: 32)
  --lora-target-modules M    comma-separated modules to attach LoRA to
                             (default: q_proj,k_proj,v_proj,o_proj,gate_proj,
                             up_proj,down_proj)
  --no-4bit                  load the base model in 16-bit; the default is
                             4-bit quantized, which is what fits an 8B model
                             on a consumer GPU

WAFL:
  --mode {wafl,self}         'wafl' exchanges and aggregates adapters on
                             contact; 'self' never exchanges and trains every
                             node every round, as a self-training-only
                             baseline (default: wafl)
  --split-type {noniid,iid}  how the training data is spread over the nodes
                             (default: noniid)
  --n-device N_DEVICE        number of nodes in the ad hoc network (default: 10)
  --contact-file FILE        JSON file holding the node-to-node contact
                             pattern; ignored, and not required, when
                             --mode self
                             (default: ./contact_pattern/rwp_n10_a0500_r100_p10_s01.json)
  --rounds ROUNDS            number of rounds; under --mode wafl this is
                             capped by the length of the contact pattern file
                             (default: 300, which is past convergence for the
                             default RWP pattern)
  --fl-coefficiency LAMBDA   WAFL aggregation coefficient lambda. 0 disables
                             aggregation while keeping the contact schedule,
                             which is the like-for-like ablation of model
                             exchange (default: 1.0)

local training:
  --self-train-epochs E      epochs of pre-self-training before the WAFL
                             rounds (default: 0.05)
  --local-train-epochs E     epochs of local training per WAFL round
                             (default: 0.01)
  --batch-size BATCH_SIZE    micro batch size (default: 2)
  --grad-acc GRAD_ACC        gradient accumulation steps; one optimizer step
                             consumes batch-size x grad-acc samples (default: 2)
  --lr LR                    learning rate of the per-node AdamW optimizer
                             (default: 2e-4)

data and output:
  --max-samples-per-node N   cap the samples per node (default: use them all)
  --cache-dir CACHE_DIR      where MultiWOZ 2.4 is downloaded to
                             (default: ./mwz24_data)
  --output-dir OUTPUT_DIR    output directory (default:
                             ./<mode>-<model>-dst-<split-type>, with -lam<x>
                             appended when --fl-coefficiency is not 1.0)
  --save-every SAVE_EVERY    save a checkpoint of every node each N rounds;
                             0 saves only at the end. Round 0, the state right
                             after pre-self-training, and the final round are
                             always saved (default: 50)
  --seed SEED                random seed (default: 1)
```

Epochs, not steps, size the local training on purpose: under a Non-IID split
the nodes hold very different amounts of data, and equal step counts would give
the data-rich nodes proportionally less exposure to their own data than the
data-poor ones. `--local-train-epochs 0.01` means each scheduled node consumes
1 % of its local dataset per round, picking up where it left off in the previous
round rather than restarting.

### Output layout

```
wafl-qwen3-8b-dst-noniid/
├── wafl_meta.json          # mode, base model and settings behind these checkpoints
├── train_effort.json       # optimizer steps and rounds actually run per node
├── round_0000/             # after pre-self-training, before any exchange
│   ├── node_0_lora.pt      # LoRA state dict, one file per node
│   └── ...
├── round_0050/
├── round_0100/
├── ...
├── node_0/                 # final adapter in save_pretrained format
│   ├── adapter_model.safetensors
│   └── ...
└── ...
```

The `round_*` checkpoints are what the evaluation reads, and they are also what
lets you draw a convergence curve afterwards. The `node_*` directories hold the
final adapters in the standard PEFT format, ready to load anywhere.

**Round 0 is the self-training-only baseline.** It is written as soon as
pre-self-training finishes, before a single model has been exchanged, and it is
always saved regardless of `--save-every`. Evaluating it tells you how each node
performs having seen nothing but its own Non-IID slice, which is the number
every later round should be compared against. On Qwen3-8B that baseline is
53.57 % mean JGA against 79.77 % after 300 rounds; see
[Convergence](#convergence) for the full curve.

**`wafl_meta.json` is not decoration.** Llama-3.1-8B and the Mistral 7B/8B
models have the same layer count, hidden size and head configuration, so their
LoRA tensors are **shape-compatible**. Loading a Ministral adapter onto Llama
raises no error at all and quietly produces nonsense. The evaluation scripts
therefore read this file and refuse to run on a mismatch:

```
Checkpoint was trained with 'unsloth/Meta-Llama-3.1-8B-Instruct' but --model
resolves to 'unsloth/Ministral-3-8B-Instruct-2512'.
These may have identical LoRA shapes, so loading would silently produce
nonsense. Pass the matching --model, or --ignore-meta if you really mean it.
```

It also means you normally do not need to pass `--model` to the evaluation
scripts at all — they pick up the model, dialect, rank and alpha from the
checkpoint.

### What a run looks like

Loading the base model and starting Phase 1 (Qwen3-8B, RTX A6000). This
particular run predates the `--self-train-epochs` default being raised to
`0.05`, so the step counts below are lower than what a default run produces
today — the shape of the log is otherwise the same:

```
Loading base model: unsloth/Qwen3-8B
==((====))==  Unsloth 2026.7.6: Fast Qwen3 patching. Transformers: 5.5.0.
   \\   /|    NVIDIA RTX A6000. Num GPUs = 1. Max memory: 44.427 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.11.0+cu128. CUDA: 8.6. CUDA Toolkit: 12.8. Triton: 3.6.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.35. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth 2026.7.6 patched 36 layers with 36 QKV layers, 36 O layers and 36 MLP layers.

===== Phase 1: Pre-self training =====
  [self-train] node 0  (9228 samples, 0.01 epochs = 23 steps)
    node0 cursor: epoch=0.010 (epoch=0, pos=92/9228)
  ...
  [self-train] node 9  (789 samples, 0.01 epochs = 1 steps)
    node9 cursor: epoch=0.005 (epoch=0, pos=4/789)
  saved round-0 (self-train only) checkpoints to ./wafl-qwen3-8b-dst-noniid/round_0000
```

Then Phase 2, one contact pattern entry per round — only the nodes with a
non-empty neighbour list train that round:

```
===== Phase 2: WAFL rounds =====

--- round 0/300 --- contact: {'0': [8], '1': [], '2': [7], '3': [], '4': [], '5': [], '6': [], '7': [2], '8': [0], '9': []}
    node0 cursor: epoch=0.020 (epoch=0, pos=184/9228)
    node2 cursor: epoch=0.019 (epoch=0, pos=176/9177)
    node7 cursor: epoch=0.017 (epoch=0, pos=32/1934)
    node8 cursor: epoch=0.020 (epoch=0, pos=16/817)
```

Nodes `1, 3, 4, 5, 6, 9` have no neighbour in this particular contact-pattern
entry, so they are skipped for both aggregation and local training that
round — exactly as described in "How it works" above.

---

## Baselines and ablations

An accuracy number for WAFL means nothing on its own. Two comparisons matter,
and they answer different questions.

### Does collaboration beat simply training longer?

`--mode self` never exchanges anything and trains **every node in every round**,
ignoring the contact pattern entirely (no contact file is needed):

```bash
uv run wafl_llm_dst_train.py --mode self
uv run wafl_llm_dst_eval.py  --mode self
```

This is deliberately generous to the baseline. Under RWP a node only meets
somebody in roughly a fifth to a quarter of rounds, so `--mode self` hands each
node about **four to five times more gradient steps** than the WAFL run gets.
If WAFL still wins, the conclusion is strong: the gain comes from exchange, not
from optimization budget, and no amount of extra local training closes the gap.

The point of the round-0 checkpoint is that it is only 0.05 epochs of
self-training, so it cannot answer this question by itself — hence the separate
run.

### Does the gain come from exchange specifically?

For a strict ablation, keep everything about the WAFL run and disable only the
aggregation, by setting the coefficient to zero:

```bash
uv run wafl_llm_dst_train.py --fl-coefficiency 0
uv run wafl_llm_dst_eval.py  --fl-coefficiency 0
```

With `lambda = 0` the aggregation step is an exact no-op, but the contact
schedule still decides who trains in which round. Every node therefore performs
**exactly the same optimizer steps on exactly the same batches** as in the real
WAFL run; the only difference in the entire experiment is whether adapters are
mixed. That is the cleanest possible attribution of the effect, and it needs no
special mode.

### Keeping the runs apart

Each configuration writes to its own directory, so nothing is overwritten:

| command | directory | summary |
| ------- | --------- | ------- |
| (defaults) | `wafl-qwen3-8b-dst-noniid/` | `wafl_qwen3-8b_eval_noniid.json` |
| `--mode self` | `self-qwen3-8b-dst-noniid/` | `self_qwen3-8b_eval_noniid.json` |
| `--fl-coefficiency 0` | `wafl-qwen3-8b-dst-noniid-lam0/` | `wafl_qwen3-8b_eval_noniid_lam0.json` |
| `--mode self --split-type iid` | `self-qwen3-8b-dst-iid/` | `self_qwen3-8b_eval_iid.json` |

The `-lam<x>` suffix appears only when the coefficient differs from 1.0, so
ordinary runs keep their existing paths. The mode is recorded in
`wafl_meta.json`, and the evaluation scripts pick it up automatically, so
`--mode` mainly matters for finding the right directory.

**Phase 1 is identical in both modes.** With the same seed, a WAFL run and a
`--mode self` run produce a byte-identical `round_0000`, so the two curves start
from the same point and diverge only because of what happens afterwards.

### Auditing the comparison

Because the modes do different amounts of work, every run prints how much
training each node actually received, and writes the same figures to
`train_effort.json`:

```
===== Local training performed (including pre-self-training) =====
  node 0:    920 optimizer steps, trained in   69 of 300 rounds
  ...
  total: 8300 optimizer steps across 10 nodes
  Compare this line between --mode wafl and --mode self before reading anything
  into an accuracy difference.
```

Under `--mode self` every node shows 300 of 300 rounds; under WAFL the count
reflects how often that node actually met someone. Quoting these numbers
alongside any accuracy comparison is what makes it checkable.

### A third comparison worth running

```bash
uv run wafl_llm_dst_train.py --mode self --split-type iid
```

Self-training on an IID split should do well, because every node already sees
every domain. If WAFL on Non-IID data approaches that number while `--mode self`
on Non-IID data does not, the effect is specific to the Non-IID setting rather
than being a generic benefit of longer training.

---

## Quantitative evaluation

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

### Options

```
model:
  --model MODEL              preset alias or Hugging Face id; defaults to
                             whatever the checkpoint metadata records, or
                             qwen3-8b if there is none
  --list-models              print the built-in model presets and exit
  --dialect DIALECT          override the prompt format (default: from metadata)
  --loader {auto,language,multimodal}
                             which unsloth loader to use (default: auto)
  --max-seq-len MAX_SEQ_LEN  maximum prompt length in tokens (default: 2048)
  --lora-rank LORA_RANK      LoRA rank r (default: from metadata, else 16)
  --lora-alpha LORA_ALPHA    LoRA alpha (default: from metadata, else 32)
  --no-4bit                  load the base model in 16-bit; the default is
                             4-bit quantized
  --no-fast-inference        generate with use_cache=False to avoid unsloth's
                             decode-only kernel; use this if generation
                             crashes inside *_fast_forward_inference with a
                             broadcast shape error (see Troubleshooting)

what to evaluate:
  --mode {wafl,self}         which run to look at: the WAFL run or the
                             self-training baseline. Only selects the default
                             output directory (default: wafl)
  --fl-coefficiency LAMBDA   aggregation coefficient of the run being
                             evaluated. Only selects the default output
                             directory (default: 1.0)
  --split-type {noniid,iid}  split type used for training, which selects the
                             default output directory (default: noniid)
  --output-dir OUTPUT_DIR    directory holding the checkpoints (default:
                             ./<mode>-<model>-dst-<split-type>, with -lam<x>
                             appended when --fl-coefficiency is not 1.0)
  --n-device N_DEVICE        number of nodes to look for (default: 10)
  --nodes NODES              comma-separated list of nodes to evaluate,
                             e.g. 0,3,7 (default: all of them)
  --round ROUND              evaluate this round number only
                             (default: the latest checkpoint)
  --rounds-curve             evaluate every saved round to obtain a
                             convergence curve; this multiplies the runtime
  --ignore-meta              proceed even if the checkpoint was produced by a
                             different base model; normally a mismatch is
                             fatal because several supported models share
                             LoRA shapes

decoding and metrics:
  --max-test MAX_TEST        number of test turns to score (default: 300)
  --max-new-tokens N         generation budget per turn (default: 256)
  --strict-json              count any output that is not pure JSON as a
                             failure; by default the outermost JSON object is
                             pulled out of the answer

output:
  --cache-dir CACHE_DIR      where MultiWOZ 2.4 is downloaded to
                             (default: ./mwz24_data)
  --out-json OUT_JSON        where to write the summary
                             (default: ./<mode>_<model>_eval_<split-type>.json)
```

The summary is also written as JSON, with one record per evaluated round and
the per-node metrics nested inside, so it can be plotted directly.

### Metrics

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

## Qualitative evaluation

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
worked, those same slots are filled in afterwards. Worked through below.

### Options

```
model:
  --model MODEL              preset alias or Hugging Face id; defaults to
                             whatever the checkpoint metadata records, or
                             qwen3-8b if there is none
  --list-models              print the built-in model presets and exit
  --dialect DIALECT          override the prompt format (default: from metadata)
  --loader {auto,language,multimodal}
                             which unsloth loader to use (default: auto)
  --max-seq-len MAX_SEQ_LEN  maximum prompt length in tokens (default: 2048)
  --lora-rank LORA_RANK      LoRA rank r (default: from metadata, else 16)
  --lora-alpha LORA_ALPHA    LoRA alpha (default: from metadata, else 32)
  --no-4bit                  load the base model in 16-bit; the default is
                             4-bit quantized
  --no-fast-inference        generate with use_cache=False to avoid unsloth's
                             decode-only kernel; use this if generation
                             crashes inside *_fast_forward_inference with a
                             broadcast shape error (see Troubleshooting)

what to inspect (round and node are required):
  --round ROUND              round checkpoint to inspect; 0 is the
                             self-training-only baseline (required)
  --node NODE                which node's adapter to inspect (required)
  --mode {wafl,self}         which run to look at: the WAFL run or the
                             self-training baseline. Only selects the default
                             output directory (default: wafl)
  --fl-coefficiency LAMBDA   aggregation coefficient of the run being
                             evaluated. Only selects the default output
                             directory (default: 1.0)
  --split-type {noniid,iid}  split type used for training, which selects the
                             default output directory (default: noniid)
  --output-dir OUTPUT_DIR    directory holding the checkpoints (default:
                             ./<mode>-<model>-dst-<split-type>, with -lam<x>
                             appended when --fl-coefficiency is not 1.0)
  --ignore-meta              proceed even if the checkpoint was produced by a
                             different base model

decoding:
  --max-test MAX_TEST        number of test turns to run, matching
                             wafl_llm_dst_eval.py (default: 300)
  --max-new-tokens N         generation budget per turn (default: 256)
  --strict-json              count any output that is not pure JSON as a
                             failure; by default the outermost JSON object is
                             pulled out of the answer

what to print:
  --max-show MAX_SHOW        how many turns to print in full; all --max-test
                             turns still count towards the summary
                             (default: 20)
  --only-errors              print only the turns that did not match exactly
  --history-turns N          how many preceding turns of the conversation to
                             show above each prediction, 0 to hide (default: 4)
  --show-raw                 also print the raw model output before parsing;
                             parse failures always show it
  --no-color                 disable ANSI colors; they are disabled
                             automatically when stdout is not a terminal

output:
  --cache-dir CACHE_DIR      where MultiWOZ 2.4 is downloaded to
                             (default: ./mwz24_data)
  --out-md OUT_MD            also write the report as Markdown to this file,
                             convenient for pasting into notes or an issue
  --out-jsonl OUT_JSONL      also write one JSON record per turn to this file
```

The JSON parsing and the metrics are imported from `wafl_llm_dst_eval.py`
rather than reimplemented, so a turn counted as correct here is counted as
correct there too, and `--strict-json` behaves identically in both.

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
convergence curve above only shows in aggregate.

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

## Files

| File | Purpose |
| ---- | ------- |
| `wafl_llm_dst_train.py`  | WAFL training loop: pre-self-training, then rounds of exchange, aggregation and local adjustment |
| `wafl_llm_dst_eval.py`   | Quantitative: scores each node's adapter on the IID test set; JGA, slot F1, spread across nodes |
| `wafl_llm_dst_inspect.py`| Qualitative: turn-by-turn generated JSON vs. reference for one round and one node |
| `wafl_llm_data_split.py` | Splits the MultiWOZ 2.4 training dialogues over the nodes, Non-IID or IID |
| `model_registry.py`      | Model presets, per-family prompt templates, loader selection |
| `mwz24_data.py`          | Downloads MultiWOZ 2.4, extracts belief states, builds samples |
| `contact_pattern/`       | Node-to-node contact traces (copied from WAFL-MLP) |

---

## Implementation notes

* **unsloth is imported before torch, transformers and peft** in every entry
  point, because its patches to those packages only apply fully when it is
  imported first. That is why `import unsloth` sits above the other imports.
* **Adding a model** normally means one line in `MODEL_PRESETS`
  (`model_registry.py`). Adding a new model *family* means one extra entry in
  `DIALECTS` as well. Nothing else in the codebase is model-specific.
* **Vision-capable models.** Ministral 3 ships with vision capability, so its
  presets ask for unsloth's unified `FastModel` loader rather than
  `FastLanguageModel`; `--loader` overrides this. If LoRA ends up attached to
  vision-tower projections (they share names like `q_proj` with the language
  tower), narrow the list with `--lora-target-modules`. The adapter parameter
  count printed at start-up is the quickest way to notice: it should be close
  to the same figure as the other 8B presets.
* **Prompts are built as raw strings** rather than through
  `apply_chat_template`, so the exact same text is used for training and for
  generation and the data pipeline needs no tokenizer. `verify_prompt_template`
  cross-checks this against the tokenizer at start-up.
* **BOS tokens are left to the tokenizer.** Llama and Mistral tokenizers add
  theirs automatically and Qwen uses none, so the templates never write one.
* **Dialogue history holds belief states, not system utterances.** Each
  assistant entry in the history is the JSON state from that turn, so the model
  is conditioned on the state so far. `dialogue_to_samples()` in
  `mwz24_data.py` shows where to swap in the natural-language system replies
  instead.
* **Tokenization happens once**, at start-up, for every node. The training loop
  is hand-written rather than using `SFTTrainer` precisely so that no
  re-tokenization occurs on each of hundreds of rounds. Sequences are stored
  unpadded and padded per batch, which keeps host memory use low.
* **LoRA dropout is 0.** Aggregation compares parameters across nodes, and
  dropout would inject noise into that comparison.
* **Tokenizer efficiency differs between families.** The same dialogue costs a
  different number of tokens under each tokenizer, so `--max-seq-len 2048`
  truncates slightly differently and a given `--local-train-epochs` covers a
  different token budget. Worth noting when comparing families.

---

## Licensing

The code in this directory follows the repository's GNU General Public License
v3.0. The **model weights are separate** and carry their own terms:

* Qwen3 and Ministral 3 presets are **Apache-2.0**.
* Llama 3.1 / 3.2 presets are under the **Llama Community License**, which is
  not an OSI open-source licence. It carries an attribution requirement
  ("Built with Llama"), a naming rule for derivative models, and an acceptable
  use policy. If you publish adapters trained on a Llama base, those terms
  apply to them.

Choose accordingly for whatever you intend to release.

---

## References

1. H. Ochiai, Y. Sun, Q. Jin, N. Wongwiwatchai, H. Esaki,
   "Wireless Ad Hoc Federated Learning: A Fully Distributed Cooperative Machine
   Learning," 2022. <https://arxiv.org/abs/2205.11779>
2. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021.
   <https://arxiv.org/abs/2106.09685>
3. Qwen Team, "Qwen3 Technical Report," 2025.
   <https://arxiv.org/abs/2505.09388>
4. Mistral AI, "Introducing Mistral 3," 2025.
   <https://mistral.ai/news/mistral-3/>
5. Meta, "The Llama 3 Herd of Models," 2024.
   <https://arxiv.org/abs/2407.21783>
6. F. Ye, J. Manotumruksa, E. Yilmaz, "MultiWOZ 2.4: A Multi-Domain
   Task-Oriented Dialogue Dataset with Essential Annotation Corrections,"
   2021. <https://github.com/smartyfh/MultiWOZ2.4>
7. Unsloth. <https://github.com/unslothai/unsloth>

See the [repository root](../README.md) for the full WAFL publication list and
the other projects in this code space.

## License

Same as the rest of this repository: GNU General Public License v3.0. See
[LICENSE](../LICENSE).
