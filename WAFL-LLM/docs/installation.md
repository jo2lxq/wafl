# Installation

[Back to the README](../README.md)

---

Full setup instructions, including the CUDA version pinning that most installation problems come down to.

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

## Setting up

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
