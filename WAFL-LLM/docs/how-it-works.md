# How it works

[Back to the README](../README.md)

---

The training algorithm: what is exchanged, when, and how it is aggregated.

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
