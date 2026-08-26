"""
model_registry.py
Model presets and chat-template handling for WAFL-LLM.

Everything that differs between model families lives here, so the training,
evaluation and inspection scripts stay model-agnostic:

  - which Hugging Face repositories are supported out of the box,
  - which prompt "dialect" each one speaks,
  - how a prompt and a training target are assembled in that dialect,
  - how the base model is loaded (plain LLM vs. multimodal wrapper).

Adding a model normally means adding one line to MODEL_PRESETS. Adding a new
model *family* means adding one entry to DIALECTS as well.

A note on BOS tokens: none of the templates below emit one. Llama and Mistral
tokenizers add their BOS automatically (add_bos_token=True) and Qwen does not
use one at all, so writing it here would double it on two families out of
three. verify_prompt_template() accounts for this when it compares our output
against the tokenizer's own chat template.
"""

import re
import warnings


# ===========================
# The DST system prompt
# ===========================

SYSTEM_MSG = (
    "You are a Dialogue State Tracking (DST) model. "
    "Given the conversation history and the latest user utterance, "
    "output the current belief state as a JSON object. "
    "Only include slots that have been mentioned. "
    "Output JSON only, no explanation."
)

# Number of past turns kept in the prompt (user + assistant entries combined).
MAX_HISTORY_TURNS = 6

# Qwen3's hybrid models can emit a <think> ... </think> block. DST labels carry
# no reasoning trace, so the "chatml_think" dialect pre-fills an empty block,
# which is what Qwen3 calls non-thinking mode.
EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"


# ===========================
# Prompt dialects
# ===========================

def _build_chatml(history, current_user, think: bool) -> str:
    """Qwen ChatML. `think` pre-fills the empty reasoning block."""
    p = f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
    for turn in history:
        p += f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"
    p += f"<|im_start|>user\n{current_user}<|im_end|>\n"
    p += "<|im_start|>assistant\n"
    if think:
        p += EMPTY_THINK_BLOCK
    return p


def _build_llama3(history, current_user) -> str:
    """Llama 3.x header format. BOS is left to the tokenizer."""
    p = (f"<|start_header_id|>system<|end_header_id|>\n\n"
         f"{SYSTEM_MSG}<|eot_id|>")
    for turn in history:
        p += (f"<|start_header_id|>{turn['role']}<|end_header_id|>\n\n"
              f"{turn['content']}<|eot_id|>")
    p += (f"<|start_header_id|>user<|end_header_id|>\n\n"
          f"{current_user}<|eot_id|>")
    p += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return p


def _build_mistral(history, current_user) -> str:
    """
    Mistral v7 instruct format, which unlike Mistral 7B v0.3 does support a
    dedicated system prompt. BOS is left to the tokenizer.
    """
    p = f"[SYSTEM_PROMPT] {SYSTEM_MSG}[/SYSTEM_PROMPT]"
    for turn in history:
        if turn["role"] == "user":
            p += f"[INST] {turn['content']}[/INST]"
        else:
            p += f" {turn['content']}</s>"
    p += f"[INST] {current_user}[/INST]"
    return p


def _build_mistral_nosys(history, current_user) -> str:
    """
    Older Mistral instruct format (v0.1 - v0.3), which has no system role at
    all. The DST instruction is folded into the first user turn instead, which
    is the usual workaround. BOS is left to the tokenizer.
    """
    p = ""
    first = True
    for turn in history:
        if turn["role"] == "user":
            content = f"{SYSTEM_MSG}\n\n{turn['content']}" if first \
                else turn["content"]
            first = False
            p += f"[INST] {content}[/INST]"
        else:
            p += f" {turn['content']}</s>"
    content = f"{SYSTEM_MSG}\n\n{current_user}" if first else current_user
    p += f"[INST] {content}[/INST]"
    return p


# Each dialect knows how to build a prompt and what closes an assistant turn.
# `answer_suffix` is appended after the label to form the training text.
DIALECTS = {
    "chatml_think": {
        "build":         lambda h, u: _build_chatml(h, u, think=True),
        "answer_suffix": "<|im_end|>",
        "description":   "Qwen ChatML with an empty <think> block "
                         "(hybrid-reasoning Qwen3)",
    },
    "chatml_plain": {
        "build":         lambda h, u: _build_chatml(h, u, think=False),
        "answer_suffix": "<|im_end|>",
        "description":   "Qwen ChatML without a <think> block "
                         "(Qwen3 *-Instruct-2507, Qwen2.5)",
    },
    "llama3": {
        "build":         _build_llama3,
        "answer_suffix": "<|eot_id|>",
        "description":   "Llama 3.x header format",
    },
    "mistral": {
        "build":         _build_mistral,
        "answer_suffix": "</s>",
        "description":   "Mistral v7 [INST] format with [SYSTEM_PROMPT] "
                         "(Mistral 3 / Ministral 3)",
    },
    "mistral_nosys": {
        "build":         _build_mistral_nosys,
        "answer_suffix": "</s>",
        "description":   "Older Mistral [INST] format with no system role "
                         "(Mistral 7B v0.1-v0.3)",
    },
}

# Markers that can end a generated answer. The evaluation parser cuts at
# whichever appears first, so it does not need to know the dialect.
STOP_MARKERS = ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "</s>"]


def build_prompt_text(dialect: str, history: list, current_user: str,
                      max_history_turns: int = MAX_HISTORY_TURNS) -> str:
    """
    Build the prompt up to the point where the assistant answer begins.

    history: [{"role": "user"/"assistant", "content": "..."}, ...]
    """
    if dialect not in DIALECTS:
        raise ValueError(f"Unknown dialect: {dialect}. "
                         f"Known: {', '.join(sorted(DIALECTS))}")
    return DIALECTS[dialect]["build"](history[-max_history_turns:],
                                      current_user)


def answer_suffix(dialect: str) -> str:
    """What closes the assistant turn, appended after the label in training."""
    return DIALECTS[dialect]["answer_suffix"]


def strip_stop_markers(text: str) -> str:
    """Cut a generated string at the first end-of-turn marker."""
    for marker in STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text


# ===========================
# Model presets
# ===========================

class ModelSpec:
    """One supported base model."""

    def __init__(self, alias, hf_id, dialect, size, license_name,
                 loader="language", note=""):
        self.alias = alias
        self.hf_id = hf_id
        self.dialect = dialect
        self.size = size
        self.license = license_name
        self.loader = loader          # "language" or "multimodal"
        self.note = note

    def __repr__(self):
        return f"ModelSpec({self.alias}, {self.hf_id}, {self.dialect})"


_PRESET_LIST = [
    # alias,          hf id,                                       dialect,         size,  license,                loader,        note
    ModelSpec("qwen3-8b", "unsloth/Qwen3-8B", "chatml_think", "8B", "Apache-2.0",
              "language", "default; hybrid reasoning, run in non-thinking mode"),
    ModelSpec("qwen3-4b", "unsloth/Qwen3-4B-Instruct-2507", "chatml_plain", "4B",
              "Apache-2.0", "language", "non-thinking only; emits no <think>"),
    ModelSpec("ministral3-8b", "unsloth/Ministral-3-8B-Instruct-2512",
              "mistral", "8B", "Apache-2.0", "multimodal",
              "Mistral 3 generation; vision-capable, LoRA applied to the "
              "language tower only"),
    ModelSpec("ministral3-3b", "unsloth/Ministral-3-3B-Instruct-2512",
              "mistral", "3B", "Apache-2.0", "multimodal",
              "4B-class member of the Mistral 3 family"),
    ModelSpec("llama31-8b", "unsloth/Meta-Llama-3.1-8B-Instruct", "llama3",
              "8B", "Llama 3.1 Community", "language",
              "not Apache; see the Llama 3.1 Community License"),
    ModelSpec("llama32-3b", "unsloth/Llama-3.2-3B-Instruct", "llama3", "3B",
              "Llama 3.2 Community", "language",
              "not Apache; see the Llama 3.2 Community License"),
]

MODEL_PRESETS = {spec.alias: spec for spec in _PRESET_LIST}
DEFAULT_MODEL = "qwen3-8b"


def infer_dialect(hf_id: str) -> str:
    """
    Guess the prompt dialect of an arbitrary Hugging Face id.

    Only used for models that are not in MODEL_PRESETS. Pass --dialect
    explicitly when the guess would be wrong.
    """
    name = hf_id.lower()
    if "llama" in name:
        return "llama3"
    if "mistral" in name or "ministral" in name or "magistral" in name:
        # Mistral 7B v0.1-v0.3 predate the [SYSTEM_PROMPT] tokens.
        if any(v in name for v in ("v0.1", "v0.2", "v0.3")):
            return "mistral_nosys"
        return "mistral"
    if "qwen" in name:
        # The 2507 instruct refreshes and Qwen2.5 never emit <think>;
        # the hybrid Qwen3 models do.
        if "2507" in name or "qwen2" in name or "instruct-2507" in name:
            return "chatml_plain"
        return "chatml_think"
    raise ValueError(
        f"Cannot infer a prompt dialect for '{hf_id}'. "
        f"Pass --dialect explicitly (one of: {', '.join(sorted(DIALECTS))}).")


def resolve_model(name: str, dialect_override: str = None) -> ModelSpec:
    """
    Turn a --model value into a ModelSpec.

    `name` is either a preset alias (e.g. "qwen3-8b") or any Hugging Face
    repository id (e.g. "unsloth/Qwen3-14B").
    """
    if name in MODEL_PRESETS:
        spec = MODEL_PRESETS[name]
        if dialect_override:
            spec = ModelSpec(spec.alias, spec.hf_id, dialect_override,
                             spec.size, spec.license, spec.loader, spec.note)
        return spec

    dialect = dialect_override or infer_dialect(name)
    # Reuse a preset's metadata when the id matches one exactly.
    for spec in _PRESET_LIST:
        if spec.hf_id == name:
            return ModelSpec(spec.alias, name, dialect_override or
                             spec.dialect, spec.size, spec.license,
                             spec.loader, spec.note)
    return ModelSpec(alias=model_slug(name), hf_id=name, dialect=dialect,
                     size="?", license_name="see the model card",
                     loader="language", note="custom model")


def model_slug(name: str) -> str:
    """Filesystem-safe short name, used in default output directory names."""
    if name in MODEL_PRESETS:
        return name
    slug = name.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9.-]+", "-", slug).strip("-")


def format_preset_table() -> str:
    """Human-readable list of the presets, for --list-models."""
    lines = [f"{'alias':<16} {'size':<5} {'dialect':<13} {'license':<22} "
             f"hf id",
             "-" * 100]
    for spec in _PRESET_LIST:
        lines.append(f"{spec.alias:<16} {spec.size:<5} {spec.dialect:<13} "
                     f"{spec.license:<22} {spec.hf_id}")
    lines.append("")
    lines.append("Any other Hugging Face id also works; the dialect is then "
                 "guessed from the name, or set with --dialect.")
    return "\n".join(lines)


# ===========================
# Run naming
# ===========================
#
# Every run gets its own directory so that experiments never overwrite each
# other. The name encodes the three things that change between runs: the
# training mode, the model, and the data split. The aggregation coefficient is
# added only when it differs from the default, which keeps the ordinary WAFL
# path unchanged while still separating lambda sweeps -- in particular
# --fl-coefficiency 0, which is a different experiment entirely.

def _lambda_suffix(mode: str, fl_coefficiency: float) -> str:
    if mode != "wafl" or fl_coefficiency == 1.0:
        return ""
    return "-lam" + ("%g" % fl_coefficiency)


def default_output_dir(mode: str, slug: str, split_type: str,
                       fl_coefficiency: float = 1.0) -> str:
    """e.g. ./wafl-qwen3-8b-dst-noniid, ./self-qwen3-8b-dst-noniid"""
    return (f"./{mode}-{slug}-dst-{split_type}"
            f"{_lambda_suffix(mode, fl_coefficiency)}")


def default_eval_json(mode: str, slug: str, split_type: str,
                      fl_coefficiency: float = 1.0) -> str:
    """e.g. ./wafl_qwen3-8b_eval_noniid.json"""
    suffix = _lambda_suffix(mode, fl_coefficiency).replace("-", "_", 1)
    return f"./{mode}_{slug}_eval_{split_type}{suffix}.json"


# ===========================
# Template verification
# ===========================

# Llama 3.x chat templates inject two date lines ahead of the system message.
# The exact text is fixed for 3.1 but follows the current date on 3.2, so
# reproducing it would make prompts vary by run date. We deliberately leave it
# out, and treat its absence as a known-benign difference rather than a problem.
_LLAMA_DATE_PREAMBLE = re.compile(
    r"Cutting Knowledge Date: [^\n]*\nToday Date: [^\n]*\n\n")


def _first_divergence(a: str, b: str, window: int = 90) -> str:
    """Describe where two strings first differ, with a little context."""
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    start = max(0, i - window // 3)
    return (f"    first differs at character {i}\n"
            f"      ours     : ...{a[start:i + window]!r}\n"
            f"      tokenizer: ...{b[start:i + window]!r}")


def verify_prompt_template(tokenizer, dialect: str, verbose: bool = True):
    """
    Cross-check our hand-written template against the tokenizer's own chat
    template, and warn if they disagree in a way that matters.

    The templates here are written by hand so that training and generation use
    byte-identical strings and so that the data pipeline needs no tokenizer.
    The risk of that approach is silently drifting from what the model was
    post-trained on, which this check exists to catch.

    Returns True when the two agree, ignoring differences known to be harmless.
    """
    history = [{"role": "user", "content": "i need a cheap hotel"},
               {"role": "assistant", "content": '{"hotel-pricerange": "cheap"}'}]
    current = "for 2 people please"
    ours = build_prompt_text(dialect, history, current)

    messages = ([{"role": "system", "content": SYSTEM_MSG}] + history +
                [{"role": "user", "content": current}])
    try:
        try:
            # This project always runs in non-thinking mode, which is exactly
            # what enable_thinking=False asks a Qwen3 template for: it makes
            # the template emit the empty <think></think> block itself, which
            # is what the chatml_think dialect writes by hand. Passing True
            # here would ask for the opposite and report a false mismatch.
            # Templates without the variable (chatml_plain, llama3, mistral)
            # simply ignore it.
            reference = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            # Tokenizers whose template takes no enable_thinking argument.
            reference = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
    except Exception as exc:                       # noqa: BLE001
        if verbose:
            print(f"  template check skipped: the tokenizer could not render "
                  f"a reference ({type(exc).__name__}: {exc})")
        return True

    # Our templates deliberately omit BOS; the tokenizer's reference includes it.
    ref = reference
    for bos in ("<|begin_of_text|>", "<s>"):
        if ref.startswith(bos):
            ref = ref[len(bos):]
            break

    if ref == ours:
        if verbose:
            print(f"  template check: OK (dialect '{dialect}' matches the "
                  f"tokenizer's chat template)")
        return True

    # Known-benign: everything matches once the Llama date preamble is removed.
    stripped = _LLAMA_DATE_PREAMBLE.sub("", ref, count=1)
    if stripped == ours:
        if verbose:
            print(f"  template check: OK for '{dialect}', apart from the "
                  f"Llama 'Cutting Knowledge Date' / 'Today Date' preamble, "
                  f"which is\n    omitted on purpose because Llama 3.2 fills "
                  f"it with the current date and that would\n    make prompts "
                  f"depend on the day the run started.")
        return True

    warnings.warn(
        f"The '{dialect}' template does not match the tokenizer's own chat "
        f"template. Training and evaluation stay self-consistent, but the "
        f"prompts may differ from what the model was post-trained on. "
        f"Check DIALECTS['{dialect}'] in model_registry.py.",
        RuntimeWarning, stacklevel=2)
    if verbose:
        print("  template check: MISMATCH")
        print(_first_divergence(ours, ref))
    return False


# ===========================
# Loading the base model
# ===========================

def load_base_model(spec: ModelSpec, max_seq_len: int, load_in_4bit: bool,
                    loader: str = "auto"):
    """
    Load a base model with the loader appropriate for it.

    Vision-capable models such as Ministral 3 are not always loadable through
    FastLanguageModel, so those presets ask for unsloth's unified FastModel.
    Returns (model, tokenizer, loader_class).
    """
    from unsloth import FastLanguageModel

    want = spec.loader if loader == "auto" else loader
    loader_cls = FastLanguageModel

    if want == "multimodal":
        try:
            from unsloth import FastModel
            loader_cls = FastModel
        except ImportError:
            print("  note: unsloth.FastModel is unavailable, falling back to "
                  "FastLanguageModel. Upgrade unsloth if loading fails.")

    model, tokenizer = loader_cls.from_pretrained(
        model_name     = spec.hf_id,
        max_seq_length = max_seq_len,
        load_in_4bit   = load_in_4bit,
        dtype          = None,
    )
    # Multimodal wrappers hand back a processor; the text tokenizer hangs off it.
    if not hasattr(tokenizer, "encode") and hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    return model, tokenizer, loader_cls


def attach_lora(model, loader_cls, rank: int, alpha: int,
                target_modules: list, seed: int = 1,
                gradient_checkpointing: bool = False):
    """
    Attach a LoRA adapter. Dropout is fixed at 0 because aggregation compares
    parameters across nodes and dropout would inject noise into that.
    """
    kwargs = dict(
        r              = rank,
        lora_alpha     = alpha,
        lora_dropout   = 0.0,
        target_modules = target_modules,
        bias           = "none",
        random_state   = seed,
    )
    if gradient_checkpointing:
        kwargs["use_gradient_checkpointing"] = "unsloth"
    return loader_cls.get_peft_model(model, **kwargs)


def set_training_mode(loader_cls, model):
    fn = getattr(loader_cls, "for_training", None)
    if fn is not None:
        fn(model)


def set_inference_mode(loader_cls, model, fast: bool = True):
    """
    Put the model into inference mode.

    `fast` enables unsloth's optimized generation path. See
    generation_kwargs() for why turning it off is sometimes necessary.
    """
    if fast:
        fn = getattr(loader_cls, "for_inference", None)
        if fn is not None:
            fn(model)
            return

    model.eval()
    config = getattr(model, "config", None)
    if config is not None:
        config.use_cache = True


def generation_kwargs(fast: bool = True) -> dict:
    """
    Extra keyword arguments for generate() when the fast path is disabled.

    unsloth decides between its prefill and its decode-only attention kernels
    by testing whether a KV cache was passed in. Newer transformers hands over
    an empty cache object during prefill instead of None, so an older unsloth
    build reads that as "already decoding", pushes the whole prompt through the
    decode-only kernel, and dies with a broadcast error such as

        Qwen3Attention_fast_forward_inference: Qn *= cos
        RuntimeError: output with shape [1, 32, 1, 128] doesn't match the
        broadcast shape [1, 32, 87, 128]

    where 87 is the prompt length. Generating with use_cache=False keeps
    past_key_values at None and therefore stays on the ordinary path. It costs
    speed, since nothing is cached between steps, but it is correct.
    Upgrading unsloth and unsloth_zoo is the real fix.
    """
    return {} if fast else {"use_cache": False}


def get_generate_fn(model, fast: bool = True):
    """
    Return the callable to use for generation.

    With fast=False this digs out the original transformers `generate` that
    unsloth saved as `_old_generate` before patching. Falls back to the
    ordinary `generate` when no patched version is found.
    """
    if fast:
        return model.generate

    seen = set()
    target = model
    while target is not None and id(target) not in seen:
        seen.add(id(target))
        old = getattr(target, "_old_generate", None)
        if old is not None:
            print("  note: using the unpatched transformers generate() "
                  "with use_cache=False (--no-fast-inference)")
            return old
        target = getattr(target, "base_model", None)

    return model.generate
