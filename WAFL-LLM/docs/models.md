# Supported models and prompt dialects

[Back to the README](../README.md)

---

Which base models are supported, how to add more, and how each model family's chat format is handled.

```bash
uv run wafl_llm_dst_train.py --list-models
```

| Alias | Size | Hugging Face id | Prompt dialect | License |
| ----- | ---- | --------------- | -------------- | ------- |
| `qwen3-8b` *(default)* | 8B | `unsloth/Qwen3-8B` | `chatml_think` | Apache-2.0 |
| `qwen3-4b` | 4B | `unsloth/Qwen3-4B-Instruct-2507` | `chatml_plain` | Apache-2.0 |
| `ministral3-8b` | 8B | `unsloth/Ministral-3-8B-Instruct-2512` | `mistral` | Apache-2.0 |
| `ministral3-3b` | 3B | `unsloth/Ministral-3-3B-Instruct-2512` | `mistral` | Apache-2.0 |
| `llama31-8b` | 8B | `unsloth/Meta-Llama-3.1-8B-Instruct` | `llama3` | Llama 3.1 Community |
| `llama32-3b` | 3B | `unsloth/Llama-3.2-3B-Instruct` | `llama3` | Llama 3.2 Community |

Three families across two size classes, which lets you separate "does WAFL work
regardless of model family" from "how does model capacity affect convergence".
Any other Hugging Face id works too — the dialect is guessed from the name, or
set explicitly with `--dialect`.

**The LoRA payload is nearly identical across all six.** At rank 16 on the seven
attention and MLP projections, every 8B-class model here lands around 40 M
adapter parameters, under 100 MB in fp16. Since that payload *is* the
communication cost in WAFL, model families can be compared without the
comparison being confounded by how much data each one has to transmit. The
training script prints the exact figure at start-up.

### Prompt dialects

Each family formats a conversation differently, so prompt construction is
factored into `model_registry.py`:

| Dialect | Format | Used by |
| ------- | ------ | ------- |
| `chatml_think` | ChatML plus an empty `<think>` block | hybrid-reasoning Qwen3 |
| `chatml_plain` | ChatML, no `<think>` block | Qwen3 `*-Instruct-2507`, Qwen2.5 |
| `llama3` | `<\|start_header_id\|>` headers | Llama 3.x |
| `mistral` | `[INST]` with `[SYSTEM_PROMPT]` | Mistral 3 / Ministral 3 |
| `mistral_nosys` | `[INST]`, system folded into the first user turn | Mistral 7B v0.1–v0.3 |

Two details are worth knowing about:

* **Qwen3 is not one dialect.** The hybrid `Qwen3-8B` expects a `<think>` block
  and is put into non-thinking mode by pre-filling an empty one. The
  `Qwen3-4B-Instruct-2507` refresh never emits `<think>` at all, so inserting
  the block there would be wrong. Hence two ChatML dialects rather than one.
* **Older Mistral has no system role.** Mistral 7B v0.1–v0.3 reject a system
  message outright, so `mistral_nosys` folds the DST instruction into the first
  user turn. Mistral 3 / Ministral 3 support `[SYSTEM_PROMPT]` properly and use
  the plain `mistral` dialect.

The templates are written out by hand rather than going through
`apply_chat_template`, so that training and generation use byte-identical
strings and the data pipeline needs no tokenizer. The risk of that approach is
silently drifting from what the model was post-trained on, so at start-up the
training script **checks its template against the tokenizer's own** and warns on
any mismatch:

```
  template check: OK (dialect 'chatml_think' matches the tokenizer's chat template)
```

If you see `MISMATCH`, the check prints the exact character where the two
strings first diverge, with context on both sides, so you can fix
`DIALECTS[...]` in `model_registry.py`.

**One expected, benign difference.** Llama 3.x chat templates inject two lines
ahead of the system message:

```
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a Dialogue State Tracking (DST) model. ...
```

The `llama3` dialect here leaves those out on purpose, because Llama 3.2 fills
`Today Date` with the *current* date — which would make prompts depend on the
day a run happened to start, and would make a Llama run differ from the other
families in prompt content rather than just markup. Since training and
evaluation both use the same fixed prefix, LoRA absorbs it and results stay
valid. The checker recognises this specific case and reports it without warning:

```
  template check: OK for 'llama3', apart from the Llama 'Cutting Knowledge Date' /
    'Today Date' preamble, which is omitted on purpose ...
```

---
