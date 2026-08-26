"""
wafl_llm_dst_inspect.py
Qualitative inspection of a single WAFL-trained adapter.

Where wafl_llm_dst_eval.py answers "how good is the network", this script
answers "what is this one node actually doing". It runs one node's adapter from
one round over the test dialogues and prints, turn by turn, the conversation fed
to the model, the JSON it produced, the reference JSON, and a slot-level diff of
the two.

Exactly one round and one node are required, because printing every node would
bury the detail this script exists to show:

    python wafl_llm_dst_inspect.py --round 50 --node 3

The parsing and metric code is imported from wafl_llm_dst_eval.py, so a turn
counted as correct here is counted as correct there too.

Requires model_registry.py, mwz24_data.py and wafl_llm_dst_eval.py in the same
directory.
Run `python wafl_llm_dst_inspect.py --help` for the full list of options.
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
)
from mwz24_data import CACHE_DIR, load_split
from wafl_llm_dst_eval import (
    TARGET_MODULES,
    apply_metadata,
    compute_jga,
    compute_slot_f1,
    list_round_dirs,
    load_metadata,
    node_ckpt_path,
    parse_json_output,
)

# Slot-level outcome labels used throughout the report.
OK       = "OK"        # predicted value matches the reference
WRONG    = "WRONG"     # slot present on both sides, values differ
MISSING  = "MISSING"   # in the reference, not predicted
SPURIOUS = "SPURIOUS"  # predicted, not in the reference


# ===========================
# Command line options
# ===========================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Qualitative, turn-by-turn inspection of one WAFL-trained "
                    "LoRA adapter on the MultiWOZ 2.4 test set.")

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

    g = p.add_argument_group("what to inspect (round and node are required)")
    g.add_argument("--round", type=int, required=True,
                   help="round checkpoint to inspect; 0 is the "
                        "self-training-only baseline (required)")
    g.add_argument("--node", type=int, required=True,
                   help="which node's adapter to inspect (required)")
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
    g.add_argument("--ignore-meta", action="store_true",
                   help="proceed even if the checkpoint was produced by a "
                        "different base model")

    g = p.add_argument_group("decoding")
    g.add_argument("--max-test", type=int, default=300,
                   help="number of test turns to run, matching "
                        "wafl_llm_dst_eval.py (default: 300)")
    g.add_argument("--max-new-tokens", type=int, default=256,
                   help="generation budget per turn (default: 256)")
    g.add_argument("--strict-json", action="store_true",
                   help="count any output that is not pure JSON as a failure; "
                        "by default the outermost JSON object is pulled out of "
                        "the answer")

    g = p.add_argument_group("what to print")
    g.add_argument("--max-show", type=int, default=20,
                   help="how many turns to print in full; all --max-test turns "
                        "still count towards the summary (default: 20)")
    g.add_argument("--only-errors", action="store_true",
                   help="print only the turns that did not match exactly")
    g.add_argument("--history-turns", type=int, default=4,
                   help="how many preceding turns of the conversation to show "
                        "above each prediction, 0 to hide (default: 4)")
    g.add_argument("--show-raw", action="store_true",
                   help="also print the raw model output before parsing; "
                        "parse failures always show it")
    g.add_argument("--no-color", action="store_true",
                   help="disable ANSI colors; they are disabled automatically "
                        "when stdout is not a terminal")

    g = p.add_argument_group("output")
    g.add_argument("--cache-dir", default=CACHE_DIR,
                   help="where MultiWOZ 2.4 is downloaded to "
                        f"(default: {CACHE_DIR})")
    g.add_argument("--out-md", default=None,
                   help="also write the report as Markdown to this file, "
                        "convenient for pasting into notes or an issue")
    g.add_argument("--out-jsonl", default=None,
                   help="also write one JSON record per turn to this file")
    return p


def resolve_args(argv=None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.list_models:
        print(format_preset_table())
        sys.exit(0)

    args.target_modules = list(TARGET_MODULES)

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
    args.color = (not args.no_color) and sys.stdout.isatty()
    return args


# ===========================
# Small formatting helpers
# ===========================

_ANSI = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "green": "\033[32m", "red": "\033[31m",
    "yellow": "\033[33m", "cyan": "\033[36m",
}

_STATUS_COLOR = {OK: "green", WRONG: "red", MISSING: "yellow",
                 SPURIOUS: "cyan"}


def paint(text: str, color: str, enabled: bool) -> str:
    if not enabled or color not in _ANSI:
        return text
    return f"{_ANSI[color]}{text}{_ANSI['reset']}"


# One complete chat turn, in any of the supported prompt dialects. The trailing
# assistant header has no closing tag, so it is naturally excluded.
TURN_PATTERNS = [
    # ChatML (Qwen)
    re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>",
               re.DOTALL),
    # Llama 3.x
    re.compile(r"<\|start_header_id\|>(system|user|assistant)"
               r"<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", re.DOTALL),
]

# Mistral needs its own pass, because its markers do not name the speaker.
_MISTRAL_RE = re.compile(r"\[INST\](.*?)\[/INST\]([^\[]*)", re.DOTALL)


def prompt_to_turns(prompt: str) -> list:
    """
    Recover the conversation from a prompt string as [(role, content), ...],
    dropping the system message. The last entry is the current user utterance.
    """
    for pattern in TURN_PATTERNS:
        found = pattern.findall(prompt)
        if found:
            return [(role, content.strip()) for role, content in found
                    if role != "system"]

    turns = []
    for user_part, assistant_part in _MISTRAL_RE.findall(prompt):
        turns.append(("user", user_part.strip()))
        answer = assistant_part.replace("</s>", "").strip()
        if answer:
            turns.append(("assistant", answer))
    return turns


def shorten(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


# ===========================
# Slot-level comparison
# ===========================

def diff_states(pred: dict, gold: dict) -> list:
    """
    Compare a predicted belief state against the reference, slot by slot.
    Returns [(status, slot, predicted_value, gold_value), ...] sorted by slot.
    """
    rows = []
    for slot in sorted(set(pred) | set(gold)):
        p = pred.get(slot)
        g = gold.get(slot)
        if slot in pred and slot in gold:
            status = OK if p == g else WRONG
        elif slot in gold:
            status = MISSING
        else:
            status = SPURIOUS
        rows.append((status, slot, p, g))
    return rows


# ===========================
# Report rendering
# ===========================

def render_turn_text(idx: int, sample: dict, pred: dict, raw: str,
                     parse_ok: bool, rows: list, exact: bool, cfg) -> str:
    """Render one turn as coloured plain text."""
    c = cfg.color
    out = []

    verdict = (paint("EXACT MATCH", "green", c) if exact
               else paint("MISMATCH", "red", c))
    head = f"--- turn {idx} --- {verdict}"
    if not parse_ok:
        head += "  " + paint("[PARSE FAILED]", "red", c)
    out.append(paint(head, "bold", c))

    turns = prompt_to_turns(sample["prompt"])
    history, current = turns[:-1], (turns[-1] if turns else ("user", ""))
    if cfg.history_turns > 0 and history:
        out.append(paint("  conversation so far:", "dim", c))
        for role, content in history[-cfg.history_turns:]:
            label = "user " if role == "user" else "state"
            out.append(paint(f"    {label} | {shorten(content)}", "dim", c))

    out.append(f"  {paint('USER', 'bold', c)}  | {current[1]}")

    if rows:
        out.append("  slots:")
        for status, slot, p, g in rows:
            col = _STATUS_COLOR[status]
            mark = paint(f"{status:<8}", col, c)
            if status == OK:
                out.append(f"    {mark} {slot} = {p}")
            elif status == WRONG:
                out.append(f"    {mark} {slot}: predicted {p!r}, gold {g!r}")
            elif status == MISSING:
                out.append(f"    {mark} {slot} = {g!r}  (not predicted)")
            else:
                out.append(f"    {mark} {slot} = {p!r}  (not in reference)")
    else:
        out.append(paint("  slots: both sides empty", "dim", c))

    if cfg.show_raw or not parse_ok:
        out.append(paint(f"  raw output: {shorten(raw, 400)}", "dim", c))

    return "\n".join(out)


def render_turn_md(idx: int, sample: dict, pred: dict, raw: str,
                   parse_ok: bool, rows: list, exact: bool, cfg) -> str:
    """Render one turn as Markdown."""
    out = []
    verdict = "EXACT MATCH" if exact else "MISMATCH"
    if not parse_ok:
        verdict += " (parse failed)"
    out.append(f"### Turn {idx} - {verdict}\n")

    turns = prompt_to_turns(sample["prompt"])
    history, current = turns[:-1], (turns[-1] if turns else ("user", ""))
    if cfg.history_turns > 0 and history:
        out.append("Conversation so far:\n")
        for role, content in history[-cfg.history_turns:]:
            label = "user" if role == "user" else "state"
            out.append(f"- `{label}` {shorten(content)}")
        out.append("")

    out.append(f"**User:** {current[1]}\n")
    out.append("| | slot | predicted | gold |")
    out.append("| --- | --- | --- | --- |")
    if rows:
        for status, slot, p, g in rows:
            pv = "-" if p is None else f"`{p}`"
            gv = "-" if g is None else f"`{g}`"
            out.append(f"| {status} | `{slot}` | {pv} | {gv} |")
    else:
        out.append("| OK | _(empty state)_ | - | - |")
    out.append("")

    if cfg.show_raw or not parse_ok:
        out.append(f"Raw output:\n\n```\n{shorten(raw, 400)}\n```\n")
    return "\n".join(out)


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

    # --- Locate the requested checkpoint ---
    all_rounds = list_round_dirs(cfg.output_dir)
    if not all_rounds:
        raise FileNotFoundError(
            f"No round_* checkpoints under {cfg.output_dir}. "
            "Run wafl_llm_dst_train.py first, or pass --output-dir.")

    match = [d for r, d in all_rounds if r == cfg.round]
    if not match:
        available = ", ".join(str(r) for r, _ in all_rounds)
        raise FileNotFoundError(
            f"Round {cfg.round} not found. Available rounds: {available}")

    pt_path = node_ckpt_path(match[0], cfg.node)
    if pt_path is None:
        raise FileNotFoundError(
            f"No adapter for node {cfg.node} in {match[0]}.")
    print(f"\nInspecting round {cfg.round}, node {cfg.node}\n  {pt_path}")

    # --- Model, adapter, data ---
    print(f"\nLoading base model: {cfg.spec.hf_id}")
    model, tokenizer, loader_cls = load_base_model(
        cfg.spec, cfg.max_seq_len, cfg.load_in_4bit, cfg.loader)
    model = attach_lora(model, loader_cls, cfg.lora_rank, cfg.lora_alpha,
                        cfg.target_modules)
    set_inference_mode(loader_cls, model, cfg.fast_inference)
    generate_fn = get_generate_fn(model, cfg.fast_inference)
    gen_kwargs = generation_kwargs(cfg.fast_inference)

    device = next(model.parameters()).device
    params = torch.load(pt_path, map_location="cpu")
    set_peft_model_state_dict(model, {k: v.to(device)
                                      for k, v in params.items()})

    test_samples = load_split("test", cfg.spec.dialect,
                              max_samples=cfg.max_test,
                              cache_dir=cfg.cache_dir)

    # --- Run the turns ---
    predictions, labels = [], []
    records = []
    n_parse_fail = 0
    slot_counts = {OK: 0, WRONG: 0, MISSING: 0, SPURIOUS: 0}

    for sample in tqdm(test_samples, desc=f"r{cfg.round}/node{cfg.node}",
                       unit="turn", dynamic_ncols=True):
        inputs = tokenizer(
            sample["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_seq_len,
        ).to(model.device)

        outputs = generate_fn(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            **gen_kwargs,
        )
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=False)

        pred, parse_ok = parse_json_output(raw, strict=cfg.strict_json)
        if not parse_ok:
            n_parse_fail += 1

        gold = sample["gold"]
        rows = diff_states(pred, gold)
        for status, *_ in rows:
            slot_counts[status] += 1

        predictions.append(pred)
        labels.append(gold)
        records.append({"sample": sample, "pred": pred, "raw": raw,
                        "parse_ok": parse_ok, "rows": rows,
                        "exact": pred == gold})

    # --- Print the turn-by-turn report ---
    shown = [(i, r) for i, r in enumerate(records)
             if not (cfg.only_errors and r["exact"])]
    clipped = shown[:cfg.max_show]

    print("\n" + "=" * 68)
    what = "mismatching turns" if cfg.only_errors else "turns"
    print(f"Qualitative report - {cfg.slug}, round {cfg.round}, "
          f"node {cfg.node} (showing {len(clipped)} of {len(shown)} {what})")
    print("=" * 68)
    for i, r in clipped:
        print()
        print(render_turn_text(i, r["sample"], r["pred"], r["raw"],
                               r["parse_ok"], r["rows"], r["exact"], cfg))
    if len(shown) > len(clipped):
        print(f"\n... {len(shown) - len(clipped)} more not shown "
              f"(raise --max-show to see them)")

    # --- Summary over every turn that was run ---
    jga = compute_jga(predictions, labels)
    f1 = compute_slot_f1(predictions, labels)
    n = len(test_samples)

    print("\n" + "=" * 68)
    print(f"Summary over all {n} turns "
          f"({cfg.slug}, round {cfg.round}, node {cfg.node})")
    print("=" * 68)
    print(f"Exact-match turns : {sum(1 for r in records if r['exact'])}/{n} "
          f"(JGA {jga * 100:.2f}%)")
    print(f"Slot precision    : {f1['precision'] * 100:.2f}%")
    print(f"Slot recall       : {f1['recall'] * 100:.2f}%")
    print(f"Slot F1           : {f1['f1'] * 100:.2f}%")
    print(f"Parse failures    : {n_parse_fail} ({n_parse_fail / n * 100:.1f}%)")
    print("Slot outcomes     : "
          f"{slot_counts[OK]} ok, {slot_counts[WRONG]} wrong value, "
          f"{slot_counts[MISSING]} missing, {slot_counts[SPURIOUS]} spurious")
    print("=" * 68)
    print("MISSING dominating suggests under-prediction, SPURIOUS dominating "
          "suggests the\nmodel is inventing slots, WRONG dominating means it "
          "finds the slot but not the value.")

    # --- Optional files ---
    if cfg.out_md:
        with open(cfg.out_md, "w", encoding="utf-8") as f:
            f.write(f"# Qualitative report - round {cfg.round}, "
                    f"node {cfg.node}\n\n")
            f.write(f"- Model: `{cfg.spec.hf_id}` (dialect "
                    f"`{cfg.spec.dialect}`)\n")
            f.write(f"- Checkpoint: `{pt_path}`\n")
            f.write(f"- Turns run: {n}\n")
            f.write(f"- JGA: {jga * 100:.2f}%  |  "
                    f"Slot F1: {f1['f1'] * 100:.2f}%  |  "
                    f"Parse failures: {n_parse_fail}\n")
            f.write(f"- Slot outcomes: {slot_counts[OK]} ok, "
                    f"{slot_counts[WRONG]} wrong, {slot_counts[MISSING]} "
                    f"missing, {slot_counts[SPURIOUS]} spurious\n\n")
            for i, r in clipped:
                f.write(render_turn_md(i, r["sample"], r["pred"], r["raw"],
                                       r["parse_ok"], r["rows"], r["exact"],
                                       cfg))
                f.write("\n")
        print(f"\nMarkdown report written to {cfg.out_md}")

    if cfg.out_jsonl:
        with open(cfg.out_jsonl, "w", encoding="utf-8") as f:
            for i, r in enumerate(records):
                f.write(json.dumps({
                    "turn": i,
                    "model": cfg.spec.hf_id,
                    "round": cfg.round,
                    "node": cfg.node,
                    "utterance": r["sample"]["utterance"],
                    "pred": r["pred"],
                    "gold": r["sample"]["gold"],
                    "exact": r["exact"],
                    "parse_ok": r["parse_ok"],
                    "raw": r["raw"],
                }, ensure_ascii=False) + "\n")
        print(f"Per-turn records written to {cfg.out_jsonl}")


if __name__ == "__main__":
    main()
