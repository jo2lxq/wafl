# WAFL-LLM

Wireless Ad Hoc Federated Learning (WAFL) of an **LLM LoRA adapter** for
**Dialogue State Tracking** on MultiWOZ 2.4.

Ten simulated devices each hold a different slice of dialogue data. Each trains
a LoRA adapter locally and, whenever the mobility pattern brings two of them
within radio range, they exchange and average **only their LoRA parameters**.
No parameter server, no global model, no coordinator.

The point is that a device which only ever saw taxi bookings ends up able to
track hotel and restaurant states too. On Qwen3-8B, collaboration takes mean
Joint Goal Accuracy from **53.6 % to 79.8 %**, and the spread between devices
falls roughly sevenfold.

This is the WAFL-MLP algorithm carried from a small MLP on MNIST to an
instruction-tuned LLM on a structured language task.

---

## Quick start

Needs an NVIDIA GPU. A 4-bit 8B model with LoRA fits comfortably on 24 GB; the
3B and 4B presets need much less.

```bash
uv init
uv python pin 3.12
uv add torch torchvision unsloth datasets transformers trl peft accelerate bitsandbytes tqdm
```

`uv add torch` can pick a CUDA build newer than your driver supports, which
fails in confusing ways. If `torch.cuda.is_available()` comes back `False`, see
[docs/installation.md](docs/installation.md) — it is a two-line fix in
`pyproject.toml`.

Get a contact pattern:

```bash
mkdir -p contact_pattern
cp ../WAFL-MLP/data/contact_pattern/rwp_n10_a0500_r100_p10_s01.json contact_pattern/
```

Train, then evaluate. MultiWOZ 2.4 downloads itself on first run.

```bash
uv run wafl_llm_dst_train.py          # 10 nodes, 300 rounds, Non-IID split
uv run wafl_llm_dst_eval.py           # scores every node on the same test set
```

Look at what one node actually predicts, turn by turn:

```bash
uv run wafl_llm_dst_inspect.py --round 300 --node 9 --only-errors
```

## Choosing a model

```bash
uv run wafl_llm_dst_train.py --model ministral3-8b
```

| Alias | Size | Hugging Face id | License |
| ----- | ---- | --------------- | ------- |
| `qwen3-8b` *(default)* | 8B | `unsloth/Qwen3-8B` | Apache-2.0 |
| `qwen3-4b` | 4B | `unsloth/Qwen3-4B-Instruct-2507` | Apache-2.0 |
| `ministral3-8b` | 8B | `unsloth/Ministral-3-8B-Instruct-2512` | Apache-2.0 |
| `ministral3-3b` | 3B | `unsloth/Ministral-3-3B-Instruct-2512` | Apache-2.0 |
| `llama31-8b` | 8B | `unsloth/Meta-Llama-3.1-8B-Instruct` | Llama 3.1 Community |
| `llama32-3b` | 3B | `unsloth/Llama-3.2-3B-Instruct` | Llama 3.2 Community |

Any other Hugging Face id works too. Each model writes to its own directory, so
runs never overwrite each other. See [docs/models.md](docs/models.md).

## Baselines

A WAFL number means little on its own. Two comparisons matter:

```bash
uv run wafl_llm_dst_train.py --mode self            # never exchanges; trains every node every round
uv run wafl_llm_dst_train.py --fl-coefficiency 0    # keeps the contact schedule, exchanges nothing
```

The first asks whether collaboration beats simply training longer; the second
is the strict ablation of exchange itself. Details in
[docs/experiments.md](docs/experiments.md).

## Results

Qwen3-8B, Non-IID split, 10 nodes, 300 test turns:

| round | mean JGA | std JGA | mean slot F1 |
| ----- | -------- | ------- | ------------ |
| 0 (self-training only) | 53.57 % | 13.93 % | 91.01 % |
| 150 | 80.57 % | 1.19 % | 97.92 % |
| 300 | 79.77 % | 1.89 % | 97.78 % |

The falling standard deviation is the WAFL-specific result: the nodes converge
on a shared model instead of staying ten local specialists. The `taxi` nodes,
with fewer than 1000 samples each, gain nearly 49 points and finish level with
nodes holding ten times more data.

Full tables, the per-node breakdown and the error analysis are in
[docs/results.md](docs/results.md).

## Documentation

| | |
| --- | --- |
| [Background](docs/background.md) | why collaborative DST tuning, and why DST |
| [How it works](docs/how-it-works.md) | the algorithm, phases and contact patterns |
| [Supported models](docs/models.md) | presets, prompt dialects, adding a model |
| [Data split](docs/data-split.md) | how MultiWOZ 2.4 is divided over the nodes |
| [Installation](docs/installation.md) | full setup, CUDA pinning, requirements |
| [Baselines and ablations](docs/experiments.md) | the comparisons worth running |
| [Results](docs/results.md) | measured numbers, convergence, error analysis |
| [Command line reference](docs/cli.md) | every option of every script |
| [Troubleshooting](docs/troubleshooting.md) | errors that have actually come up |
| [Implementation notes](docs/implementation-notes.md) | design decisions, licensing, references |

There is also an [index of all documentation](docs/README.md).

## Files

| File | Purpose |
| ---- | ------- |
| `wafl_llm_dst_train.py` | training: pre-self-training, then rounds of exchange and local adjustment |
| `wafl_llm_dst_eval.py` | quantitative: JGA and slot F1 for every node |
| `wafl_llm_dst_inspect.py` | qualitative: predicted vs reference state, turn by turn |
| `wafl_llm_data_split.py` | splits MultiWOZ 2.4 over the nodes, Non-IID or IID |
| `model_registry.py` | model presets, prompt templates, run naming |
| `mwz24_data.py` | downloads MultiWOZ 2.4, builds prompts and labels |

## License

GNU General Public License v3.0, as with the rest of this repository. Model
weights carry their own terms; the Llama presets are not Apache-licensed. See
[docs/implementation-notes.md](docs/implementation-notes.md).
