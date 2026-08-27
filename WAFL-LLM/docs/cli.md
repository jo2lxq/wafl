# Command line reference

[Back to the README](../README.md)

---

Every option of every script. `--help` prints the same thing.

## wafl_llm_dst_train.py

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


Epochs, not steps, size the local training on purpose: under a Non-IID split
the nodes hold very different amounts of data, and equal step counts would give
the data-rich nodes proportionally less exposure to their own data than the
data-poor ones. `--local-train-epochs 0.01` means each scheduled node consumes
1 % of its local dataset per round, picking up where it left off in the previous
round rather than restarting.
```

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
[docs/results.md](results.md) for the full curve.

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
round — exactly as described in [how-it-works.md](how-it-works.md).

---

## wafl_llm_dst_eval.py

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
                             broadcast shape error (see troubleshooting.md)

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


The summary is also written as JSON, with one record per evaluated round and
the per-node metrics nested inside, so it can be plotted directly.
```


## wafl_llm_dst_inspect.py

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
                             broadcast shape error (see troubleshooting.md)

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


The JSON parsing and the metrics are imported from `wafl_llm_dst_eval.py`
rather than reimplemented, so a turn counted as correct here is counted as
correct there too, and `--strict-json` behaves identically in both.
```
