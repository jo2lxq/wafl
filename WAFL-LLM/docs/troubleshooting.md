# Troubleshooting

[Back to the README](../README.md)

---

Problems that have actually come up, and what they mean.

**Generation crashes with a broadcast shape error.** If evaluation dies inside
unsloth's attention kernel like this:

```
File ".../unsloth/models/qwen3.py", line 313, in Qwen3Attention_fast_forward_inference
    Qn *= cos
RuntimeError: output with shape [1, 32, 1, 128] doesn't match the broadcast
shape [1, 32, 87, 128]
```

then unsloth is older than the installed `transformers`. The `1` in the first
shape is a single decode step and the `87` is the prompt length: unsloth has
routed the whole prompt through its decode-only kernel. It picks between its
prefill and decode kernels by testing whether a KV cache was handed in, and
newer transformers passes an empty cache object during prefill instead of
`None`, which older unsloth builds read as "already decoding".

Upgrade both unsloth packages, which are versioned as a pair:

```bash
uv add -U unsloth unsloth_zoo
uv run python -c "import unsloth; print(unsloth.__version__)"
```

LoRA weights do not depend on the library version, so **existing checkpoints
stay valid** and there is no need to retrain.

If the upgrade is not possible — newer GPUs can pin you to a CUDA build that
only an older unsloth supports — generate with the cache disabled instead:

```bash
uv run wafl_llm_dst_eval.py --no-fast-inference
```

This passes `use_cache=False`, which keeps `past_key_values` at `None` and so
stays off the broken path. It is slower, since nothing is cached between steps,
but the results are identical. The same flag works on
`wafl_llm_dst_inspect.py`.

**Only some models are affected.** unsloth ships hand-written inference kernels
for Qwen and Llama, so those hit the bug; Ministral 3 loads through `FastModel`
and falls back to stock transformers, which is why it evaluates cleanly on the
same broken install. If you need the fallback, use it for every model so the
comparison stays like-for-like.

**`Unsloth should be imported before [transformers, peft]`.** unsloth patches
classes inside transformers and peft as it is imported, so importing it late
costs speed and GPU memory, though not correctness. Every entry point already
does `import unsloth` above its other imports, so this should not appear. If it
does, check that nothing was reordered above that line in the script you are
running.

**Training and evaluation may use different unsloth versions** without any
problem, since only adapter tensors are stored. It is still worth keeping them
aligned so that tokenization and prompt handling stay identical.

---
