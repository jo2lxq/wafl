"""
wafl_llm_dst_train.py
Wireless Ad Hoc Federated Learning (WAFL) of an LLM LoRA adapter for
Dialogue State Tracking (DST) on MultiWOZ 2.4.

This is the WAFL-MLP algorithm carried over to LoRA + an instruction-tuned LLM
+ MultiWOZ 2.4. Every node trains a LoRA adapter on its own Non-IID slice of
the data and, whenever the contact pattern says it meets a neighbour, exchanges
and aggregates **only the LoRA adapter parameters** with that neighbour.

Differences from WAFL-MLP:
  - exchanged object : the whole model -> the LoRA adapter parameters only
  - model            : a small MLP     -> an 8B/4B-class LLM + LoRA
  - data             : MNIST, Non-IID by label
                       -> MultiWOZ 2.4, Non-IID by dialogue domain

Several model families are supported; see `--list-models`. The prompt format
of each is handled in model_registry.py.

Required in the same directory:
  model_registry.py, mwz24_data.py, wafl_llm_data_split.py
Required contact pattern file (see README):
  contact_pattern/rwp_n10_a0500_r100_p10_s01.json

Run `python wafl_llm_dst_train.py --help` for the full list of options.
"""

import argparse
import copy
import json
import os
import sys

# unsloth patches transformers and peft as it is imported, so it must come
# before them or some of its optimizations are silently skipped.
import unsloth  # noqa: F401

import torch
from tqdm import tqdm
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from model_registry import (
    DEFAULT_MODEL,
    DIALECTS,
    default_output_dir,
    attach_lora,
    format_preset_table,
    load_base_model,
    model_slug,
    resolve_model,
    set_training_mode,
    verify_prompt_template,
)
from mwz24_data import CACHE_DIR, to_text
from wafl_llm_data_split import get_split

# LoRA is applied to all attention and MLP projections. On vision-capable
# models these names also exist inside the vision tower; --lora-target-modules
# is there so the list can be narrowed if that becomes a problem.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj",
                  "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULT_CONTACT_FILE = "./contact_pattern/rwp_n10_a0500_r100_p10_s01.json"

# Written next to the checkpoints so that evaluation can tell which base model
# produced them. This matters more than it looks: several supported models have
# identical LoRA tensor shapes, so mixing them up would not raise any error.
META_FILENAME = "wafl_meta.json"


# ===========================
# Command line options
# ===========================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WAFL training of an LLM LoRA adapter for DST "
                    "on MultiWOZ 2.4.")

    g = p.add_argument_group("model")
    g.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"preset alias or any Hugging Face id "
                        f"(default: {DEFAULT_MODEL}); see --list-models")
    g.add_argument("--list-models", action="store_true",
                   help="print the built-in model presets and exit")
    g.add_argument("--dialect", default=None, choices=sorted(DIALECTS),
                   help="override the prompt format; only needed for models "
                        "outside the preset list whose name is ambiguous")
    g.add_argument("--loader", default="auto",
                   choices=["auto", "language", "multimodal"],
                   help="which unsloth loader to use; 'auto' follows the "
                        "preset (default: auto)")
    g.add_argument("--max-seq-len", type=int, default=2048,
                   help="maximum sequence length in tokens (default: 2048)")
    g.add_argument("--lora-rank", type=int, default=16,
                   help="LoRA rank r (default: 16)")
    g.add_argument("--lora-alpha", type=int, default=32,
                   help="LoRA scaling factor alpha (default: 32)")
    g.add_argument("--lora-target-modules", default=",".join(TARGET_MODULES),
                   help="comma-separated modules to attach LoRA to "
                        f"(default: {','.join(TARGET_MODULES)})")
    g.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                   help="load the base model in 16-bit; the default is 4-bit "
                        "quantized, which is what fits an 8B model on a "
                        "consumer GPU")
    g.set_defaults(load_in_4bit=True)

    g = p.add_argument_group("WAFL")
    g.add_argument("--mode", default="wafl", choices=["wafl", "self"],
                   help="'wafl' exchanges and aggregates adapters on contact; "
                        "'self' never exchanges and trains every node every "
                        "round, as a self-training-only baseline "
                        "(default: wafl)")
    g.add_argument("--split-type", default="noniid", choices=["noniid", "iid"],
                   help="how the training data is spread over the nodes "
                        "(default: noniid)")
    g.add_argument("--n-device", type=int, default=10,
                   help="number of nodes in the ad hoc network (default: 10)")
    g.add_argument("--contact-file", default=DEFAULT_CONTACT_FILE,
                   help="JSON file holding the node-to-node contact pattern; "
                        "ignored, and not required, when --mode self "
                        f"(default: {DEFAULT_CONTACT_FILE})")
    g.add_argument("--rounds", type=int, default=300,
                   help="number of rounds; under --mode wafl this is capped by "
                        "the length of the contact pattern file (default: 300, "
                        "which is past convergence for the default RWP "
                        "pattern)")
    g.add_argument("--fl-coefficiency", type=float, default=1.0,
                   help="WAFL aggregation coefficient lambda. 0 disables "
                        "aggregation while keeping the contact schedule, which "
                        "is the like-for-like ablation of model exchange "
                        "(default: 1.0)")

    g = p.add_argument_group("local training")
    g.add_argument("--self-train-epochs", type=float, default=0.05,
                   help="epochs of pre-self-training before the WAFL rounds "
                        "(default: 0.05)")
    g.add_argument("--local-train-epochs", type=float, default=0.01,
                   help="epochs of local training per WAFL round "
                        "(default: 0.01)")
    g.add_argument("--batch-size", type=int, default=2,
                   help="micro batch size (default: 2)")
    g.add_argument("--grad-acc", type=int, default=2,
                   help="gradient accumulation steps; one optimizer step "
                        "consumes batch-size x grad-acc samples (default: 2)")
    g.add_argument("--lr", type=float, default=2e-4,
                   help="learning rate of the per-node AdamW optimizer "
                        "(default: 2e-4)")

    g = p.add_argument_group("data and output")
    g.add_argument("--max-samples-per-node", type=int, default=None,
                   help="cap the samples per node (default: use them all)")
    g.add_argument("--cache-dir", default=CACHE_DIR,
                   help="where MultiWOZ 2.4 is downloaded to "
                        f"(default: {CACHE_DIR})")
    g.add_argument("--output-dir", default=None,
                   help="output directory (default: "
                        "./<mode>-<model>-dst-<split-type>, with -lam<x> "
                        "appended when --fl-coefficiency is not 1.0)")
    g.add_argument("--save-every", type=int, default=50,
                   help="save a checkpoint of every node each N rounds; "
                        "0 saves only at the end. Round 0, the state right "
                        "after pre-self-training, and the final round are "
                        "always saved (default: 50)")
    g.add_argument("--seed", type=int, default=1,
                   help="random seed (default: 1)")
    return p


def resolve_args(argv=None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.list_models:
        print(format_preset_table())
        sys.exit(0)

    args.spec = resolve_model(args.model, args.dialect)
    args.slug = model_slug(args.model)
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.mode, args.slug,
                                             args.split_type,
                                             args.fl_coefficiency)
    args.target_modules = [m.strip() for m in args.lora_target_modules.split(",")
                           if m.strip()]
    # Autodetect bf16 support; fall back to fp16 on older GPUs.
    args.use_bf16 = bool(torch.cuda.is_available()
                         and torch.cuda.is_bf16_supported())
    return args


# ===========================
# Helpers
# ===========================

def calc_steps(n_samples: int, epochs: float, cfg) -> int:
    """
    Convert a number of epochs into a number of optimizer steps.

    One step consumes batch_size x grad_acc samples. Epochs rather than steps
    are used to size the local training because the nodes hold different
    amounts of data under a Non-IID split, and epochs keep them comparable.
    """
    effective_batch = cfg.batch_size * cfg.grad_acc
    return int(max(1, (n_samples * epochs) // effective_batch))


class DataCursor:
    """
    A cursor that walks repeatedly over one node's pre-tokenized samples.

    All samples are tokenized once at start-up and kept as variable-length
    Python lists. Avoiding padding="max_length" here cuts host memory use by a
    large factor; padding happens per batch, up to the longest sequence in that
    batch only.

    Usage:
        cursor = DataCursor(texts, tokenizer, cfg, seed=1)
        ids, mask = cursor.next(steps)
    """

    def __init__(self, texts: list, tokenizer, cfg, seed: int = 1):
        self.n = len(texts)
        if self.n == 0:
            raise ValueError(
                "A node was given zero samples. Reduce --n-device or check "
                "the data split.")
        self.cfg = cfg
        self.seed = seed
        self.epoch = 0
        self.pos = 0
        self.pad_id = (tokenizer.pad_token_id
                       if tokenizer.pad_token_id is not None else 0)
        self._order = self._new_order()

        print(f"    Pre-tokenizing {self.n} samples...", end="", flush=True)
        enc = tokenizer(
            texts,
            truncation     = True,
            max_length     = cfg.max_seq_len,
            padding        = False,   # pad per batch instead
            return_tensors = None,    # keep plain Python lists
        )
        self.input_ids_list      = enc["input_ids"]       # List[List[int]]
        self.attention_mask_list = enc["attention_mask"]  # List[List[int]]
        print(" done.")

    def _new_order(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        return torch.randperm(self.n, generator=rng).tolist()

    def next(self, steps: int):
        """
        Collect enough samples for `steps` optimizer steps and return them
        padded to the longest sequence in the batch. Reaching the end of an
        epoch simply continues into a freshly shuffled next epoch.
        """
        effective_batch = self.cfg.batch_size * self.cfg.grad_acc
        n_samples_needed = steps * effective_batch
        indices = []

        while len(indices) < n_samples_needed:
            remaining = self.n - self.pos
            needed = n_samples_needed - len(indices)
            if needed <= remaining:
                indices.extend(self._order[self.pos: self.pos + needed])
                self.pos += needed
            else:
                indices.extend(self._order[self.pos:])
                self.epoch += 1
                self.pos = 0
                self._order = self._new_order()

        batch_ids  = [self.input_ids_list[i]      for i in indices]
        batch_mask = [self.attention_mask_list[i] for i in indices]
        max_len    = max(len(x) for x in batch_ids)

        ids_padded = torch.tensor(
            [x + [self.pad_id] * (max_len - len(x)) for x in batch_ids],
            dtype=torch.long,
        )
        mask_padded = torch.tensor(
            [x + [0] * (max_len - len(x)) for x in batch_mask],
            dtype=torch.long,
        )
        return ids_padded, mask_padded

    def status(self) -> str:
        return (f"epoch={self.epoch + self.pos / self.n:.3f} "
                f"(epoch={self.epoch}, pos={self.pos}/{self.n})")


# ===========================
# Exchange and aggregation of LoRA adapters
# ===========================

def get_lora_params(model) -> dict:
    """Copy the LoRA adapter parameters out of the model onto the host."""
    sd = get_peft_model_state_dict(model)
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def set_lora_params(model, params: dict):
    """Write a set of LoRA adapter parameters back into the model."""
    device = next(model.parameters()).device
    params_on_device = {k: v.to(device) for k, v in params.items()}
    set_peft_model_state_dict(model, params_on_device)


def wafl_aggregate(local_params: dict, recv_params_list: list,
                   coeff: float) -> dict:
    """
    The WAFL aggregation rule, identical to main.py of WAFL-MLP but applied to
    LoRA parameters instead of the full model:

        update_k = recv_k - local          (difference to each neighbour)
        local   += sum_k update_k * lambda / (n_nbr + 1)

    local_params:     LoRA parameters currently held by this node
    recv_params_list: LoRA parameters received from the neighbours
    coeff:            aggregation coefficient lambda
    Returns the aggregated LoRA parameters.

    coeff == 0 is returned untouched rather than computed, both to save the
    work and to make the intent explicit: it is the ablation in which nodes
    keep the WAFL contact schedule but exchange nothing.
    """
    n_nbr = len(recv_params_list)
    if n_nbr == 0 or coeff == 0.0:
        return local_params

    aggregated = {k: v.clone() for k, v in local_params.items()}

    for recv in recv_params_list:
        for key in aggregated:
            diff = recv[key] - local_params[key]
            aggregated[key] += diff * coeff / (n_nbr + 1)

    return aggregated


# ===========================
# Local training
# ===========================

# One optimizer per node, so that the AdamW moment estimates survive across
# rounds instead of being reset every time the node is scheduled.
_node_optimizers: dict = {}


def local_train(model, loader_cls, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, max_steps: int, node_id: int,
                cfg):
    """
    Train the LoRA adapter on local data for a fixed number of steps.

    A hand-written loop is used instead of SFTTrainer so that tokenization
    happens exactly once, at start-up, rather than on every round.

    input_ids / attention_mask: the tensors returned by DataCursor.next()
    node_id: key under which this node's optimizer state is kept
    """
    set_training_mode(loader_cls, model)
    device = next(model.parameters()).device

    if node_id not in _node_optimizers:
        _node_optimizers[node_id] = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
        )
    optimizer = _node_optimizers[node_id]

    dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16

    model.train()
    optimizer.zero_grad()

    n_samples = input_ids.shape[0]
    step = 0
    sample_idx = 0

    pbar = tqdm(total=max_steps, desc=f"node{node_id}", leave=False,
                unit="step", dynamic_ncols=True)

    while step < max_steps:
        step_loss = 0.0
        # batch_size x grad_acc samples make up one optimizer step.
        for _ in range(cfg.grad_acc):
            end = min(sample_idx + cfg.batch_size, n_samples)
            batch_ids  = input_ids[sample_idx:end].to(device)
            batch_mask = attention_mask[sample_idx:end].to(device)
            sample_idx = end

            # Causal-LM labels: padding positions are excluded from the loss.
            labels = batch_ids.clone()
            labels[batch_mask == 0] = -100

            with torch.autocast(device_type=device.type, dtype=dtype):
                outputs = model(
                    input_ids      = batch_ids,
                    attention_mask = batch_mask,
                    labels         = labels,
                )
                loss = outputs.loss / cfg.grad_acc

            loss.backward()
            step_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()
        step += 1
        pbar.set_postfix(loss=f"{step_loss:.4f}")
        pbar.update(1)

    pbar.close()
    return model


# ===========================
# Checkpointing
# ===========================

def save_round_checkpoints(node_params: list, output_dir: str,
                           round_number: int) -> str:
    """
    Write every node's LoRA parameters into output_dir/round_<n>/.

    Round 0 is the state right after pre-self-training, i.e. before any model
    exchange has happened, which serves as the self-training-only baseline of
    the convergence curve.
    """
    ckpt_dir = os.path.join(output_dir, f"round_{round_number:04d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    for n, params in enumerate(node_params):
        torch.save(params, os.path.join(ckpt_dir, f"node_{n}_lora.pt"))
    return ckpt_dir


def write_metadata(cfg):
    """
    Record which base model and settings produced these checkpoints.

    Llama-3.1-8B and the Mistral 7B/8B models have identical hidden sizes,
    layer counts and head counts, so their LoRA tensors have identical shapes.
    Loading one family's adapter onto another would therefore succeed silently
    and produce nonsense. The evaluation scripts refuse to run on a mismatch
    unless told otherwise.
    """
    meta = {
        "mode":           cfg.mode,
        "model":          cfg.spec.hf_id,
        "model_alias":    cfg.spec.alias,
        "dialect":        cfg.spec.dialect,
        "lora_rank":      cfg.lora_rank,
        "lora_alpha":     cfg.lora_alpha,
        "target_modules": cfg.target_modules,
        "split_type":     cfg.split_type,
        "n_device":       cfg.n_device,
        "max_seq_len":    cfg.max_seq_len,
        "contact_file":   cfg.contact_file if cfg.mode == "wafl" else None,
        "fl_coefficiency": cfg.fl_coefficiency,
        "rounds":         cfg.rounds,
        "seed":           cfg.seed,
    }
    path = os.path.join(cfg.output_dir, META_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


# ===========================
# Main
# ===========================

def main():
    cfg = resolve_args()
    torch.random.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("===== Configuration =====")
    for key, value in sorted(vars(cfg).items()):
        if key not in ("spec",):
            print(f"  {key}: {value}")
    print(f"  run mode:   {cfg.mode}"
          + ("  (no model exchange; every node trains every round)"
             if cfg.mode == "self" else ""))
    print(f"  base model: {cfg.spec.hf_id}")
    print(f"  dialect:    {cfg.spec.dialect} "
          f"({DIALECTS[cfg.spec.dialect]['description']})")
    print(f"  license:    {cfg.spec.license}")

    # --- Split the training data over the nodes ---
    print(f"\nSplitting data: {cfg.split_type}")
    node_samples = get_split(cfg.split_type, cfg.spec.dialect,
                             n_device=cfg.n_device, seed=cfg.seed,
                             cache_dir=cfg.cache_dir)

    if cfg.max_samples_per_node:
        node_samples = [s[:cfg.max_samples_per_node] for s in node_samples]

    # --- Load the contact pattern (WAFL mode only) ---
    contact_list = None
    if cfg.mode == "wafl":
        print(f"Loading contact pattern: {cfg.contact_file}")
        if not os.path.exists(cfg.contact_file):
            raise FileNotFoundError(
                f"Contact pattern not found: {cfg.contact_file}\n"
                "Copy one from WAFL-MLP (data/contact_pattern/) into "
                "./contact_pattern/, or pass --contact-file.")
        with open(cfg.contact_file) as f:
            contact_list = json.load(f)
    else:
        print("Mode 'self': no model exchange, so no contact pattern is used.")
        if cfg.contact_file != DEFAULT_CONTACT_FILE:
            print("  note: --contact-file is ignored under --mode self")

    # --- One shared base model for all nodes ---
    #   To keep memory use low there is a single copy of the frozen base model
    #   and a single LoRA module. The state of a node is just its dictionary
    #   of LoRA parameters, which is swapped in whenever that node runs.
    print(f"\nLoading base model: {cfg.spec.hf_id}")
    model, tokenizer, loader_cls = load_base_model(
        cfg.spec, cfg.max_seq_len, cfg.load_in_4bit, cfg.loader)

    # Confirm our hand-written prompt format matches this tokenizer's own.
    verify_prompt_template(tokenizer, cfg.spec.dialect)

    # dropout stays at 0 inside attach_lora so aggregation is not perturbed
    model = attach_lora(model, loader_cls, cfg.lora_rank, cfg.lora_alpha,
                        cfg.target_modules, seed=cfg.seed,
                        gradient_checkpointing=True)

    meta_path = write_metadata(cfg)
    print(f"Wrote {meta_path}")

    # Tokenize every node's dataset once, now that the tokenizer is known.
    print("\nPre-tokenizing all node datasets...")
    node_cursors = []
    for n in range(cfg.n_device):
        texts = [to_text(s, cfg.spec.dialect) for s in node_samples[n]]
        node_cursors.append(DataCursor(texts, tokenizer, cfg,
                                       seed=cfg.seed + n))
    print("Pre-tokenization complete.")

    # Every node starts from the same freshly initialized adapter.
    init_params = get_lora_params(model)
    n_lora = sum(v.numel() for v in init_params.values())
    size_note = f"{n_lora / 1e6:.1f} M ({n_lora * 2 / 1e6:.0f} MB in fp16)"
    if cfg.mode == "wafl" and cfg.fl_coefficiency != 0.0:
        # This is the payload that actually crosses the device-to-device link.
        print(f"LoRA parameters exchanged per contact: {size_note}")
    else:
        # Nothing is exchanged: --mode self has no contacts at all, and
        # lambda = 0 keeps the contact schedule but aggregates nothing.
        reason = ("no exchange in self mode" if cfg.mode == "self"
                  else "no exchange at lambda = 0")
        print(f"LoRA parameters per node: {size_note}; {reason}")
    node_params = [copy.deepcopy(init_params) for _ in range(cfg.n_device)]

    # Optimizer steps and rounds actually performed per node. Under WAFL a node
    # only trains when it meets someone, so these counters are what makes a
    # comparison against --mode self auditable rather than assumed.
    node_steps  = [0] * cfg.n_device
    node_rounds = [0] * cfg.n_device

    # =====================================================
    # Phase 1: pre-self-training, each node on its own data
    # =====================================================
    # Identical in both modes, so with the same seed the two runs share a
    # byte-identical round 0 and diverge only from there.
    print("\n===== Phase 1: Pre-self training =====")
    for n in range(cfg.n_device):
        n_samples_n = node_cursors[n].n
        steps = calc_steps(n_samples_n, cfg.self_train_epochs, cfg)
        print(f"  [self-train] node {n}  "
              f"({n_samples_n} samples, {cfg.self_train_epochs} epochs "
              f"= {steps} steps)")
        set_lora_params(model, node_params[n])
        ids, mask = node_cursors[n].next(steps)
        local_train(model, loader_cls, ids, mask, steps, node_id=n, cfg=cfg)
        node_steps[n] += steps
        print(f"    node{n} cursor: {node_cursors[n].status()}")
        node_params[n] = get_lora_params(model)

    # Checkpoint round 0: pre-self-training finished, no exchange yet.
    # This is the self-training-only baseline for the convergence curve, so it
    # is always saved regardless of --save-every.
    ckpt_dir = save_round_checkpoints(node_params, cfg.output_dir, 0)
    print(f"  saved round-0 (self-train only) checkpoints to {ckpt_dir}")

    # =====================================================
    # Phase 2: rounds
    # =====================================================
    if cfg.mode == "wafl":
        print("\n===== Phase 2: WAFL rounds =====")
        n_rounds = min(cfg.rounds, len(contact_list))
        if n_rounds < cfg.rounds:
            print(f"  note: the contact pattern only has {len(contact_list)} "
                  f"entries, so {n_rounds} rounds will be run")
        if cfg.fl_coefficiency == 0.0:
            print("  note: lambda is 0, so nodes keep the contact schedule but "
                  "exchange nothing (ablation)")
    else:
        print("\n===== Phase 2: self-training rounds (no exchange) =====")
        n_rounds = cfg.rounds
        print(f"  every node trains in all {n_rounds} rounds")

    for rnd in range(n_rounds):
        if cfg.mode == "wafl":
            contact = contact_list[rnd]
            print(f"\n--- round {rnd}/{n_rounds} --- contact: {contact}")

            # (a) Receive the neighbours' adapters and aggregate them.
            new_params = [None] * cfg.n_device
            for n in range(cfg.n_device):
                nbr = contact[str(n)]
                recv = [node_params[k] for k in nbr]
                new_params[n] = wafl_aggregate(node_params[n], recv,
                                               cfg.fl_coefficiency)

            # Apply the aggregation results simultaneously.
            for n in range(cfg.n_device):
                if len(contact[str(n)]) > 0:
                    node_params[n] = new_params[n]

            # (b) Only nodes that met somebody adjust the aggregate on local
            #     data. A node that was alone skips training, because training
            #     alone just overfits it back onto its own Non-IID slice.
            active = [n for n in range(cfg.n_device)
                      if len(contact[str(n)]) > 0]
        else:
            # No exchange and no contact schedule: every node simply keeps
            # training on its own data. This is the generous baseline -- it
            # gives each node strictly more gradient steps than WAFL does.
            print(f"\n--- round {rnd}/{n_rounds} --- self-training "
                  f"all {cfg.n_device} nodes")
            active = list(range(cfg.n_device))

        for n in active:
            steps = calc_steps(node_cursors[n].n, cfg.local_train_epochs, cfg)
            set_lora_params(model, node_params[n])
            ids, mask = node_cursors[n].next(steps)
            local_train(model, loader_cls, ids, mask, steps, node_id=n,
                        cfg=cfg)
            node_steps[n] += steps
            node_rounds[n] += 1
            print(f"    node{n} cursor: {node_cursors[n].status()}")
            node_params[n] = get_lora_params(model)

        # Periodic checkpoints, which are also what the evaluation reads.
        if (cfg.save_every > 0 and (rnd + 1) % cfg.save_every == 0) \
                or (rnd + 1) == n_rounds:
            ckpt_dir = save_round_checkpoints(node_params, cfg.output_dir,
                                              rnd + 1)
            print(f"  saved checkpoints to {ckpt_dir}")

    # =====================================================
    # How much training each node actually received
    # =====================================================
    print("\n===== Local training performed "
          "(including pre-self-training) =====")
    for n in range(cfg.n_device):
        print(f"  node {n}: {node_steps[n]:6d} optimizer steps, trained in "
              f"{node_rounds[n]:4d} of {n_rounds} rounds")
    total = sum(node_steps)
    print(f"  total: {total} optimizer steps across {cfg.n_device} nodes")
    print("  Compare this line between --mode wafl and --mode self before "
          "reading anything\n  into an accuracy difference.")
    with open(os.path.join(cfg.output_dir, "train_effort.json"), "w",
              encoding="utf-8") as f:
        json.dump({"mode": cfg.mode, "n_rounds": n_rounds,
                   "steps_per_node": node_steps,
                   "rounds_per_node": node_rounds,
                   "total_steps": total}, f, indent=2)

    # =====================================================
    # Save the final adapter of every node
    # =====================================================
    print("\n===== Saving final adapters =====")
    for n in range(cfg.n_device):
        node_dir = os.path.join(cfg.output_dir, f"node_{n}")
        os.makedirs(node_dir, exist_ok=True)
        set_lora_params(model, node_params[n])
        model.save_pretrained(node_dir)
        tokenizer.save_pretrained(node_dir)
        print(f"  node {n} -> {node_dir}")

    print(f"\nDone. Adapters saved under {cfg.output_dir}")
    eval_flags = "" if cfg.mode == "wafl" else " --mode self"
    if cfg.mode == "wafl" and cfg.fl_coefficiency != 1.0:
        eval_flags = f" --fl-coefficiency {cfg.fl_coefficiency:g}"
    print(f"Next: uv run wafl_llm_dst_eval.py{eval_flags}")


if __name__ == "__main__":
    main()
