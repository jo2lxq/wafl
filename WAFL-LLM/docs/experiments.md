# Baselines and ablations

[Back to the README](../README.md)

---

The comparisons that make a WAFL result meaningful, and how to keep their outputs apart.

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
