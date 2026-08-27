# Implementation notes

[Back to the README](../README.md)

---

Design decisions worth knowing about before modifying the code.

* **unsloth is imported before torch, transformers and peft** in every entry
  point, because its patches to those packages only apply fully when it is
  imported first. That is why `import unsloth` sits above the other imports.
* **Adding a model** normally means one line in `MODEL_PRESETS`
  (`model_registry.py`). Adding a new model *family* means one extra entry in
  `DIALECTS` as well. Nothing else in the codebase is model-specific.
* **Vision-capable models.** Ministral 3 ships with vision capability, so its
  presets ask for unsloth's unified `FastModel` loader rather than
  `FastLanguageModel`; `--loader` overrides this. If LoRA ends up attached to
  vision-tower projections (they share names like `q_proj` with the language
  tower), narrow the list with `--lora-target-modules`. The adapter parameter
  count printed at start-up is the quickest way to notice: it should be close
  to the same figure as the other 8B presets.
* **Prompts are built as raw strings** rather than through
  `apply_chat_template`, so the exact same text is used for training and for
  generation and the data pipeline needs no tokenizer. `verify_prompt_template`
  cross-checks this against the tokenizer at start-up.
* **BOS tokens are left to the tokenizer.** Llama and Mistral tokenizers add
  theirs automatically and Qwen uses none, so the templates never write one.
* **Dialogue history holds belief states, not system utterances.** Each
  assistant entry in the history is the JSON state from that turn, so the model
  is conditioned on the state so far. `dialogue_to_samples()` in
  `mwz24_data.py` shows where to swap in the natural-language system replies
  instead.
* **Tokenization happens once**, at start-up, for every node. The training loop
  is hand-written rather than using `SFTTrainer` precisely so that no
  re-tokenization occurs on each of hundreds of rounds. Sequences are stored
  unpadded and padded per batch, which keeps host memory use low.
* **LoRA dropout is 0.** Aggregation compares parameters across nodes, and
  dropout would inject noise into that comparison.
* **Tokenizer efficiency differs between families.** The same dialogue costs a
  different number of tokens under each tokenizer, so `--max-seq-len 2048`
  truncates slightly differently and a given `--local-train-epochs` covers a
  different token budget. Worth noting when comparing families.

---

## Licensing

The code in this directory follows the repository's GNU General Public License
v3.0. The **model weights are separate** and carry their own terms:

* Qwen3 and Ministral 3 presets are **Apache-2.0**.
* Llama 3.1 / 3.2 presets are under the **Llama Community License**, which is
  not an OSI open-source licence. It carries an attribution requirement
  ("Built with Llama"), a naming rule for derivative models, and an acceptable
  use policy. If you publish adapters trained on a Llama base, those terms
  apply to them.

Choose accordingly for whatever you intend to release.

---

## References

1. H. Ochiai, Y. Sun, Q. Jin, N. Wongwiwatchai, H. Esaki,
   "Wireless Ad Hoc Federated Learning: A Fully Distributed Cooperative Machine
   Learning," 2022. <https://arxiv.org/abs/2205.11779>
2. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021.
   <https://arxiv.org/abs/2106.09685>
3. Qwen Team, "Qwen3 Technical Report," 2025.
   <https://arxiv.org/abs/2505.09388>
4. Mistral AI, "Introducing Mistral 3," 2025.
   <https://mistral.ai/news/mistral-3/>
5. Meta, "The Llama 3 Herd of Models," 2024.
   <https://arxiv.org/abs/2407.21783>
6. F. Ye, J. Manotumruksa, E. Yilmaz, "MultiWOZ 2.4: A Multi-Domain
   Task-Oriented Dialogue Dataset with Essential Annotation Corrections,"
   2021. <https://github.com/smartyfh/MultiWOZ2.4>
7. Unsloth. <https://github.com/unslothai/unsloth>

See the [repository root](../../README.md) for the full WAFL publication list
and the other projects in this code space.
