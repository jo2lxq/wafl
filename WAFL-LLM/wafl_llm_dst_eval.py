"""
wafl_llm_dst_eval.py
Evaluate the per-node LoRA adapters produced by WAFL training on the
MultiWOZ 2.4 test set.

The point of WAFL is that nodes which only ever trained on a Non-IID slice of
the data still generalize to IID data, thanks to the models they exchanged.
Every node is therefore scored on the *same* test set, which mixes all domains,
and what we look for is a high score with little spread between the nodes.

Passing --rounds-curve evaluates every saved round checkpoint instead of just
the last one, which gives the accuracy-versus-round convergence curve.

For a qualitative, turn-by-turn look at what a single node actually predicts,
see the companion script wafl_llm_dst_inspect.py.

Requires model_registry.py and mwz24_data.py in the same directory.
Run `python wafl_llm_dst_eval.py --help` for the full list of options.
"""

import argparse
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore", message=".*max_new_tokens.*")

# unsloth patches transformers and peft as it is imported, so it must come
# before them or some of its optimizations are silently skipped.
import unsloth  # noqa: F401

import torch
from tqdm import tqdm
from peft import set_peft_model_state_dict

from model_registry import (
    DEFAULT_MODEL,
    DIALECTS,
    attach_lora,
    default_eval_json,
    default_output_dir,
    format_preset_table,
    generation_kwargs,
    get_generate_fn,
    load_base_model,
    model_slug,
    resolve_model,
    set_inference_mode,
    strip_stop_markers,
)
from mwz24_data import CACHE_DIR, load_split

TARGET_MODULES = ["q_proj", "k_proj", "v_proj",
                  "o_proj", "gate_proj", "up_proj", "down_proj"]

META_FILENAME = "wafl_meta.json"


# ===========================
# Command line options
# ===========================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate WAFL-trained LLM LoRA adapters for DST "
                    "on the MultiWOZ 2.4 test set.")

    g = p.add_argument_group("model")
    g.add_argument("--model", default=None,
                   help="preset alias or Hugging Face id; defaults to "
                        "whatever the checkpoint metadata records, or "
                        f"{DEFAULT_MODEL} if there is none")
    g.add_argument("--list-models", action="store_true",
                   help="print the built-in model presets and exit")
    g.add_argument("--dialect", default=None, choices=sorted(DIALECTS),
                   help="override the prompt format (default: from metadata)")
    g.add_argument("--loader", default="auto",
                   choices=["auto", "language", "multimodal"],
                   help="which unsloth loader to use (default: auto)")
    g.add_argument("--max-seq-len", type=int, default=2048,
                   help="maximum prompt length in tokens (default: 2048)")
    g.add_argument("--lora-rank", type=int, default=None,
                   help="LoRA rank r (default: from metadata, else 16)")
    g.add_argument("--lora-alpha", type=int, default=None,
                   help="LoRA alpha (default: from metadata, else 32)")
    g.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                   help="load the base model in 16-bit; the default is 4-bit "
                        "quantized")
    g.add_argument("--no-fast-inference", dest="fast_inference",
                   action="store_false",
                   help="bypass unsloth's optimized generation path; use this "
                        "if generation crashes inside "
                        "*_fast_forward_inference with a broadcast shape "
                        "error, which indicates unsloth is older than the "
                        "installed transformers")
    g.set_defaults(load_in_4bit=True, fast_inference=True)

    g = p.add_argument_group("what to evaluate")
    g.add_argument("--mode", default="wafl", choices=["wafl", "self"],
                   help="which run to look at: the WAFL run or the "
                        "self-training baseline. Only selects the default "
                        "output directory (default: wafl)")
    g.add_argument("--fl-coefficiency", type=float, default=1.0,
                   help="aggregation coefficient of the run being evaluated. "
                        "Only selects the default output directory, which "
                        "gains a -lam<x> suffix when this is not 1.0 "
                        "(default: 1.0)")
    g.add_argument("--split-type", default="noniid", choices=["noniid", "iid"],
                   help="split type used for training, which selects the "
                        "default output directory (default: noniid)")
    g.add_argument("--output-dir", default=None,
                   help="directory holding the checkpoints (default: "
                        "./<mode>-<model>-dst-<split-type>, with -lam<x> "
                        "appended when --fl-coefficiency is not 1.0)")
    g.add_argument("--n-device", type=int, default=10,
                   help="number of nodes to look for (default: 10)")
    g.add_argument("--nodes", default=None,
                   help="comma-separated list of nodes to evaluate, "
                        "e.g. 0,3,7 (default: all of them)")
    g.add_argument("--round", type=int, default=None,
                   help="evaluate this round number only "
                        "(default: the latest checkpoint)")
    g.add_argument("--rounds-curve", action="store_true",
                   help="evaluate every saved round to obtain a convergence "
                        "curve; this multiplies the runtime")
    g.add_argument("--ignore-meta", action="store_true",
                   help="proceed even if the checkpoint was produced by a "
                        "different base model; normally a mismatch is fatal "
                        "because several supported models share LoRA shapes")

    g = p.add_argument_group("decoding and metrics")
    g.add_argument("--max-test", type=int, default=300,
                   help="number of test turns to score (default: 300)")
    g.add_argument("--max-new-tokens", type=int, default=256,
                   help="generation budget per turn (default: 256)")
    g.add_argument("--strict-json", action="store_true",
                   help="count any output that is not pure JSON as a failure; "
                        "by default the outermost JSON object is pulled out of "
                        "the answer")

    g = p.add_argument_group("output")
    g.add_argument("--cache-dir", default=CACHE_DIR,
                   help="where MultiWOZ 2.4 is downloaded to "
                        f"(default: {CACHE_DIR})")
    g.add_argument("--out-json", default=None,
                   help="where to write the summary "
                        "(default: ./<mode>_<model>_eval_<split-type>.json)")
    return p


def load_metadata(output_dir: str):
    """Read wafl_meta.json from a checkpoint directory, if it exists."""
    path = os.path.join(output_dir, META_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_metadata(args, meta, ignore: bool):
    """
    Fill in unspecified options from the checkpoint metadata, and refuse to run
    on a base-model mismatch.

    This guard exists because Llama-3.1-8B and the Mistral 7B/8B models have
    the same layer count, hidden size and head configuration, which makes their
    LoRA tensors shape-compatible. Loading one onto the other raises no error
    and quietly produces garbage.
    """
    if meta is None:
        if args.model is None:
            args.model = DEFAULT_MODEL
        return args

    if args.model is None:
        args.model = meta.get("model_alias") or meta.get("model")
    else:
        wanted = resolve_model(args.model, args.dialect)
        recorded = meta.get("model")
        if recorded and wanted.hf_id != recorded:
            msg = (f"Checkpoint was trained with '{recorded}' but --model "
                   f"resolves to '{wanted.hf_id}'.")
            if not ignore:
                raise SystemExit(
                    msg + "\nThese may have identical LoRA shapes, so loading "
                          "would silently produce nonsense. Pass the matching "
                          "--model, or --ignore-meta if you really mean it.")
            print(f"  WARNING: {msg} Continuing because --ignore-meta was "
                  f"given.")

    # The checkpoint knows how it was produced; trust it for labelling, since
    # evaluation itself is identical either way.
    if meta.get("mode"):
        if meta["mode"] != args.mode:
            print(f"  note: this checkpoint was trained with "
                  f"--mode {meta['mode']}; labelling results accordingly")
        args.mode = meta["mode"]
    if meta.get("fl_coefficiency") is not None:
        args.fl_coefficiency = meta["fl_coefficiency"]
    if args.dialect is None and meta.get("dialect"):
        args.dialect = meta["dialect"]
    if args.lora_rank is None and meta.get("lora_rank"):
        args.lora_rank = meta["lora_rank"]
    if args.lora_alpha is None and meta.get("lora_alpha"):
        args.lora_alpha = meta["lora_alpha"]
    if meta.get("target_modules"):
        args.target_modules = meta["target_modules"]
    return args


def resolve_args(argv=None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.list_models:
        print(format_preset_table())
        sys.exit(0)

    args.target_modules = list(TARGET_MODULES)

    # The output directory depends on the model, and the model may come from
    # the metadata inside that directory, so resolve in two passes.
    if args.output_dir is None:
        probe = args.model or DEFAULT_MODEL
        args.output_dir = default_output_dir(args.mode, model_slug(probe),
                                             args.split_type,
                                             args.fl_coefficiency)

    meta = load_metadata(args.output_dir)
    args = apply_metadata(args, meta, args.ignore_meta)

    args.spec = resolve_model(args.model, args.dialect)
    args.slug = model_slug(args.model)
    if args.lora_rank is None:
        args.lora_rank = 16
    if args.lora_alpha is None:
        args.lora_alpha = 32
    if args.out_json is None:
        args.out_json = default_eval_json(args.mode, args.slug,
                                          args.split_type,
                                          args.fl_coefficiency)
    if args.nodes:
        args.node_list = [int(x) for x in args.nodes.split(",") if x.strip()]
    else:
        args.node_list = list(range(args.n_device))
    return args


# ===========================
# Parsing and metrics
# ===========================

def parse_json_output(text: str, strict: bool = False):
    """
    Turn a generated string into a belief-state dict.

    Returns (state, ok). ok is False when nothing parseable came back, which
    is tracked separately because an empty belief state is a perfectly valid
    label at the start of a dialogue.

    Handles every supported family: the text is cut at whichever end-of-turn
    marker appears, and any <think> block a Qwen3 hybrid model might emit is
    removed.
    """
    text = strip_stop_markers(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    text = re.sub(r"```(?:json)?", "", text).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, True
    except json.JSONDecodeError:
        pass

    if not strict:
        # Fall back to the outermost {...} span, which rescues answers that
        # carry a stray token before or after the JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed, True
            except json.JSONDecodeError:
                pass

    return {}, False


def compute_jga(predictions, labels) -> float:
    """Joint Goal Accuracy: the whole predicted state must match exactly."""
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(labels) if labels else 0.0


def compute_slot_f1(predictions, labels) -> dict:
    """Micro precision / recall / F1 over the individual slot-value pairs."""
    tp = fp = fn = 0
    for pred, gold in zip(predictions, labels):
        pred_set = set(f"{k}={v}" for k, v in pred.items())
        gold_set = set(f"{k}={v}" for k, v in gold.items())
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


# ===========================
# Evaluating one node
# ===========================

def evaluate_node(model, tokenizer, test_samples, cfg, tag: str = "",
                  generate_fn=None, gen_kwargs=None) -> dict:
    """Score the LoRA adapter currently loaded into the model."""
    predictions, labels = [], []
    n_parse_fail = 0
    generate = generate_fn if generate_fn is not None else model.generate
    extra = gen_kwargs or {}

    for sample in tqdm(test_samples, desc=tag, leave=False, unit="turn",
                       dynamic_ncols=True):
        inputs = tokenizer(
            sample["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_seq_len,
        ).to(model.device)

        outputs = generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            **extra,
        )
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

        pred, ok = parse_json_output(gen_text, strict=cfg.strict_json)
        if not ok:
            n_parse_fail += 1
        predictions.append(pred)
        labels.append(sample["gold"])

    jga = compute_jga(predictions, labels)
    f1 = compute_slot_f1(predictions, labels)
    return {"jga": jga, **f1,
            "parse_fail_rate": n_parse_fail / len(test_samples)}


# ===========================
# Locating checkpoints
# ===========================

def list_round_dirs(output_dir: str) -> list:
    """Return [(round_number, path), ...] of the round_* directories, sorted."""
    if not os.path.isdir(output_dir):
        return []
    rounds = []
    for name in sorted(os.listdir(output_dir)):
        if not name.startswith("round_"):
            continue
        try:
            rnd = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        rounds.append((rnd, os.path.join(output_dir, name)))
    rounds.sort()
    return rounds


def node_ckpt_path(round_dir: str, node: int):
    """Path of one node's adapter inside a round directory, or None."""
    pt = os.path.join(round_dir, f"node_{node}_lora.pt")
    return pt if os.path.exists(pt) else None


# ===========================
# Statistics
# ===========================

def summarize(results: dict) -> dict:
    """Aggregate the per-node metrics of one round."""
    jgas = [r["jga"] for r in results.values()]
    f1s  = [r["f1"]  for r in results.values()]
    mean_jga = sum(jgas) / len(jgas)
    var_jga = sum((x - mean_jga) ** 2 for x in jgas) / len(jgas)
    return {
        "n_nodes":      len(results),
        "mean_jga":     mean_jga,
        "min_jga":      min(jgas),
        "max_jga":      max(jgas),
        "std_jga":      var_jga ** 0.5,
        "mean_slot_f1": sum(f1s) / len(f1s),
        "per_node":     {str(n): results[n] for n in sorted(results)},
    }


def print_summary(round_number, stats, cfg):
    print("\n" + "=" * 60)
    label = cfg.mode
    if cfg.mode == "wafl" and cfg.fl_coefficiency != 1.0:
        label += f"(lambda={cfg.fl_coefficiency:g})"
    print(f"WAFL-LLM DST results  ({label}, {cfg.slug}, "
          f"split={cfg.split_type}, round={round_number}, "
          f"{stats['n_nodes']} nodes)")
    print("=" * 60)
    print(f"Mean JGA    : {stats['mean_jga'] * 100:6.2f}%")
    print(f"Min  JGA    : {stats['min_jga'] * 100:6.2f}%")
    print(f"Max  JGA    : {stats['max_jga'] * 100:6.2f}%")
    print(f"Std  JGA    : {stats['std_jga'] * 100:6.2f}%"
          "   (smaller means the nodes converged together)")
    print(f"Mean SlotF1 : {stats['mean_slot_f1'] * 100:6.2f}%")
    print("=" * 60)


# ===========================
# Main
# ===========================

def main():
    cfg = resolve_args()

    print("===== Configuration =====")
    for key, value in sorted(vars(cfg).items()):
        if key not in ("spec",):
            print(f"  {key}: {value}")
    print(f"  base model: {cfg.spec.hf_id}")
    print(f"  dialect:    {cfg.spec.dialect}")

    # --- Decide which round checkpoints to evaluate ---
    all_rounds = list_round_dirs(cfg.output_dir)
    if not all_rounds:
        raise FileNotFoundError(
            f"No round_* checkpoints under {cfg.output_dir}. "
            "Run wafl_llm_dst_train.py first, or pass --output-dir.")

    if cfg.rounds_curve:
        targets = all_rounds
    elif cfg.round is not None:
        targets = [r for r in all_rounds if r[0] == cfg.round]
        if not targets:
            available = ", ".join(str(r[0]) for r in all_rounds)
            raise FileNotFoundError(
                f"Round {cfg.round} not found. Available rounds: {available}")
    else:
        targets = [all_rounds[-1]]
    print(f"\nEvaluating {len(targets)} round(s): "
          f"{', '.join(str(r[0]) for r in targets)}")

    # --- Build the base model and the LoRA structure once ---
    print(f"\nLoading base model: {cfg.spec.hf_id}")
    model, tokenizer, loader_cls = load_base_model(
        cfg.spec, cfg.max_seq_len, cfg.load_in_4bit, cfg.loader)
    model = attach_lora(model, loader_cls, cfg.lora_rank, cfg.lora_alpha,
                        cfg.target_modules)
    set_inference_mode(loader_cls, model, cfg.fast_inference)
    generate_fn = get_generate_fn(model, cfg.fast_inference)
    gen_kwargs = generation_kwargs(cfg.fast_inference)

    # --- Test data: all domains mixed, i.e. IID with respect to the nodes ---
    test_samples = load_split("test", cfg.spec.dialect,
                              max_samples=cfg.max_test,
                              cache_dir=cfg.cache_dir)

    device = next(model.parameters()).device
    all_stats = []

    for round_number, round_dir in targets:
        print(f"\n===== round {round_number} ({round_dir}) =====")
        results = {}
        for n in cfg.node_list:
            pt_path = node_ckpt_path(round_dir, n)
            if pt_path is None:
                print(f"  node {n}: checkpoint not found, skip")
                continue

            params = torch.load(pt_path, map_location="cpu")
            set_peft_model_state_dict(
                model, {k: v.to(device) for k, v in params.items()})

            res = evaluate_node(model, tokenizer, test_samples, cfg,
                                tag=f"r{round_number}/node{n}",
                                generate_fn=generate_fn,
                                gen_kwargs=gen_kwargs)
            results[n] = res
            print(f"  node {n}: JGA={res['jga'] * 100:5.2f}%  "
                  f"SlotF1={res['f1'] * 100:5.2f}%  "
                  f"parse_fail={res['parse_fail_rate'] * 100:4.1f}%")

        if not results:
            print("  no adapters evaluated for this round")
            continue

        stats = summarize(results)
        stats["round"] = round_number
        print_summary(round_number, stats, cfg)
        all_stats.append(stats)

    if not all_stats:
        print("Nothing was evaluated.")
        return

    summary = {
        "mode":        cfg.mode,
        "fl_coefficiency": cfg.fl_coefficiency,
        "model":       cfg.spec.hf_id,
        "model_alias": cfg.spec.alias,
        "dialect":     cfg.spec.dialect,
        "split_type":  cfg.split_type,
        "output_dir":  cfg.output_dir,
        "max_test":    len(test_samples),
        "strict_json": cfg.strict_json,
        "rounds":      all_stats,
    }
    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {cfg.out_json}")

    if len(all_stats) > 1:
        print("\nConvergence curve (round: mean JGA / std JGA):")
        for s in all_stats:
            print(f"  {s['round']:5d}: {s['mean_jga'] * 100:6.2f}% / "
                  f"{s['std_jga'] * 100:5.2f}%")


if __name__ == "__main__":
    main()
