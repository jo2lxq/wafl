# WAFL-LLM documentation

[Back to the project README](../README.md)

---

| | |
| --- | --- |
| [Background](background.md) | why collaborative DST tuning without a central server, and why DST is the flagship task |
| [How it works](how-it-works.md) | the algorithm: what is exchanged, when, and how it is aggregated |
| [Supported models](models.md) | the six presets, the prompt dialects behind them, and how to add your own |
| [Data split](data-split.md) | how MultiWOZ 2.4 is divided over the nodes, and the resulting distribution |
| [Installation](installation.md) | requirements, uv setup, and the CUDA pinning that most problems come down to |
| [Baselines and ablations](experiments.md) | `--mode self`, `--fl-coefficiency 0`, and keeping their outputs apart |
| [Results](results.md) | measured numbers: convergence curve, per-node gains, error analysis |
| [Command line reference](cli.md) | every option of every script, plus the output layout |
| [Troubleshooting](troubleshooting.md) | errors that have actually come up, and what they mean |
| [Implementation notes](implementation-notes.md) | design decisions, licensing, references |

## Where to start

Running it for the first time: [Installation](installation.md), then the quick
start in the [project README](../README.md).

Understanding the results: [How it works](how-it-works.md) →
[Data split](data-split.md) → [Results](results.md).

Modifying the code: [Implementation notes](implementation-notes.md) and
[Supported models](models.md).
