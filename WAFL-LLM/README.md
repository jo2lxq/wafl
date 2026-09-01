# WAFL-LLM

This project trains a large language model for **Dialogue State Tracking**
using **Wireless Ad Hoc Federated Learning (WAFL)**. Ten simulated devices
learn together by exchanging parts of their models whenever they meet, without
using any central server.

## Contents

The first four sections explain what the project does and why. The remaining
sections explain how to run it.

1. [Task: Dialogue State Tracking](#task-dialogue-state-tracking)
2. [Problem Setting: Non-IID Data Without a Central Server](#problem-setting-non-iid-data-without-a-central-server)
3. [Method: WAFL with LoRA Adapters](#method-wafl-with-lora-adapters)
4. [Summary of Results](#summary-of-results)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Training](#training)
8. [Quantitative Evaluation](#quantitative-evaluation)
9. [Qualitative Evaluation](#qualitative-evaluation)
10. [Model Selection](#model-selection)
11. [Baselines and Ablation Studies](#baselines-and-ablation-studies)
12. [Experiments with IID Setting](#experiments-with-iid-setting)
13. [Documentation](#documentation)
14. [Repository Structure](#repository-structure)

---

## Task: Dialogue State Tracking

A task-oriented dialogue system has to understand what the user wants. To do
this, it converts each user utterance into a structured record called a
**belief state**. The belief state is a set of slots and values. For example,
if the user says "I need a cheap hotel in the north for two people", the belief
state becomes the following JSON object.

```json
{"hotel-pricerange": "cheap", "hotel-area": "north", "hotel-book_people": "2"}
```

The task of producing this record for every turn of a conversation is called
**Dialogue State Tracking**, or DST. In this project, a language model reads
the conversation so far and outputs the belief state as JSON. We use the
**MultiWOZ 2.4** dataset, which contains about ten thousand English
conversations about hotels, restaurants, trains, attractions and taxis.

## Problem Setting: Non-IID Data Without a Central Server

Suppose that ten people each carry a device, and that each device records the
conversations of its owner. The conversations will not be the same on every
device. One person mostly books restaurants, another mostly asks about trains,
and another mostly calls taxis. In machine learning, data that is distributed
unevenly in this way is called **Non-IID** data. IID means independent and
identically distributed, so Non-IID simply means that each device sees a
different kind of data.

Non-IID data causes a problem. If a device learns only from its own data, it
becomes very good at one domain and very poor at all the others. A device that
has only seen taxi conversations cannot fill in hotel slots correctly, because
it has never seen them.

One common solution is federated learning, in which every device sends its
model to a central server, and the server combines them. However, this requires
someone to set up the server, keep it running, and be trusted by everyone.

**WAFL** avoids the central server completely. Instead, devices exchange their
models directly with each other when they happen to come within radio range,
for example when their owners pass each other in a building. Each device then
averages its own model with the models it received, and continues learning.
Over many such meetings, every device gradually obtains knowledge that came
from all the others.

## Method: WAFL with LoRA Adapters

Training a large language model completely would require exchanging billions of
parameters, which is not realistic over a short-range wireless link. We
therefore use **LoRA**, which stands for Low-Rank Adaptation. With LoRA, the
original model is kept frozen, and only a small number of additional
parameters, called an **adapter**, are trained. In this project the adapter has
about 40 million parameters, which is under 100 MB. Only the adapter is
exchanged between devices.

The training then proceeds in two phases.

1. **Pre-self-training.** Every device trains briefly on its own data, so that
   it learns the output format before any exchange happens.
2. **Rounds.** In each round, a file called a **contact pattern** tells the
   program which devices are within radio range of each other. Devices that
   meet exchange their adapters and average them. After that, each of those
   devices trains a little more on its own local data. Devices that met nobody
   do nothing in that round.

A detailed explanation, including the averaging formula, is given in
[docs/how-it-works.md](docs/how-it-works.md).

One note on terminology. This document says "device", because the setting we
are modelling is a group of people carrying devices. The programs and their
options use the word **node** for the same thing. The two words mean exactly
the same in this project.

## Summary of Results

The following table shows the result for Qwen3-8B on ten devices. "Round 0"
means the state before any exchange has taken place, so each device has learned
only from its own data.

| round | mean JGA | standard deviation of JGA |
| ----- | -------- | ------------------------- |
| 0 (no exchange yet) | 53.57 % | 13.93 % |
| 150 | 80.57 % | 1.19 % |
| 300 | 79.77 % | 1.89 % |

**JGA** stands for Joint Goal Accuracy. It is the percentage of turns for which
the predicted JSON object is exactly equal to the correct one. If even a single
slot is wrong or missing, that turn is counted as an error.

Two things are worth noticing in this table.

First, the average accuracy rises from about 54 % to about 80 %. This shows
that exchanging models helps a great deal.

Second, the standard deviation falls from about 14 % to about 2 %. The standard
deviation measures how much the devices differ from each other. A large value
means that some devices are much better than others. A small value means that
all devices have reached a similar level. This is the result that is specific
to WAFL, because it shows that the devices have converged to a shared model
rather than remaining ten separate specialists.

The effect is largest for the devices that hold the least data. The two devices
that mainly recorded taxi conversations have fewer than one thousand training
samples each. Their accuracy improves by almost 49 points, and they end up at
the same level as devices that hold ten times more data.

---

## Requirements

You will need a computer with an NVIDIA GPU. The 8B models require about 24 GB
of GPU memory when loaded in 4-bit precision. The 3B and 4B models require
considerably less. You will also need Python 3.12 and roughly 20 GB of free
disk space per model.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) to manage Python packages.

**Step 1.** Create the project environment.

```bash
uv init
uv python pin 3.12
```

**Step 2.** Install the packages.

```bash
uv add torch torchvision unsloth datasets transformers trl peft accelerate bitsandbytes tqdm
```

**Step 3.** Check that PyTorch can actually see your GPU.

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `True`, the installation is complete. If it prints `False`, the
version of PyTorch that was installed does not match your GPU driver. This is a
common problem and it is fixed by adding a few lines to `pyproject.toml`.
Please see [docs/installation.md](docs/installation.md), which explains the
procedure step by step.

**Step 4.** Prepare a contact pattern file. This file describes which devices
meet each other in each round. The same files are used by the other projects in
this repository, where they are distributed as a zip archive. If the archive has
not been extracted yet, extract it first.

```bash
unzip -n ../WAFL-MLP/data/contact_pattern/contact_pattern.zip \
      -d ../WAFL-MLP/data/contact_pattern/
```

The `-n` option means that existing files are never overwritten, so it is safe
to run this command even if the archive has already been extracted. You can
then confirm that the JSON files are present.

```bash
ls ../WAFL-MLP/data/contact_pattern/*.json
```

Finally, copy the file that this project uses by default into a directory
called `contact_pattern`.

```bash
mkdir -p contact_pattern
cp ../WAFL-MLP/data/contact_pattern/rwp_n10_a0500_r100_p10_s01.json contact_pattern/
```

If your copy of the repository places these files somewhere else, adjust the
paths above. What matters is that the file
`contact_pattern/rwp_n10_a0500_r100_p10_s01.json` exists inside this project.

## Training

The dataset is downloaded automatically the first time you run the program, so
no separate preparation is needed.

```bash
uv run wafl_llm_dst_train.py
```

With no options, this trains ten devices for 300 rounds on a Non-IID split,
using Qwen3-8B. Training takes a long time, so you may wish to try a short run
first in order to confirm that everything works.

```bash
uv run wafl_llm_dst_train.py --rounds 5 --max-samples-per-node 200 --save-every 5
```

### Running in the background

A full training run takes many hours. If you are working on a remote machine
over SSH, the run will normally be killed when you close the connection. To
prevent this, start the program in the background with `nohup`.

```bash
nohup uv run wafl_llm_dst_train.py > stdout.log 2> stderr.log < /dev/null &
```

Each part of this command has a purpose.

| part | meaning |
| ---- | ------- |
| `nohup` | the program keeps running after you log out |
| `> stdout.log` | normal output is written to `stdout.log` |
| `2> stderr.log` | warnings and progress bars are written to `stderr.log` |
| `< /dev/null` | the program reads no keyboard input, so it never stops to wait for you |
| `&` | the program runs in the background and the shell prompt returns immediately |

The normal output and the error output are sent to separate files on purpose.
The libraries used here print progress bars and warnings to the error output,
which would otherwise make the training log difficult to read.

You can watch the progress at any time with the following command.

```bash
tail -f stdout.log
```

Press Ctrl-C to stop watching. This stops only the `tail` command, not the
training itself. To confirm that training is still running, or to find its
process number, use `ps`.

```bash
ps aux | grep wafl_llm_dst_train
```

If you need to stop the training, use `kill` with the process number shown by
`ps`.

### Output of a training run

The trained adapters are saved at regular intervals during training. A saved
copy of this kind is called a **checkpoint**, and it lets you evaluate the
state of the devices at any point in the training, not only at the end. The
checkpoints are written into a directory named after the settings you used, for
example `wafl-qwen3-8b-dst-noniid/`. Because the directory name
includes the model and the split, different experiments never overwrite each
other.

## Quantitative Evaluation

Every device is tested on the same test data, which contains all five domains
mixed together. This is the important point of the measurement. A device that
only ever saw taxi conversations is also asked about hotels and restaurants.

```bash
uv run wafl_llm_dst_eval.py
```

This prints the accuracy of each device and their average. To see how the
accuracy changed as the rounds progressed, add the following option. Note that
this takes several times longer, because every saved checkpoint is evaluated.

```bash
uv run wafl_llm_dst_eval.py --rounds-curve
```

## Qualitative Evaluation

Accuracy numbers tell you how well a device performs, but they do not tell you
what kind of mistakes it makes. The third program shows the prediction and the
correct answer side by side, one turn at a time.

```bash
uv run wafl_llm_dst_inspect.py --round 300 --node 9 --only-errors
```

Here `--round 300` selects the checkpoint, and `--node 9` selects one of the
ten devices. Both options are required, because printing all ten devices would
produce far too much output. Each slot is classified into one of four
categories, which makes it easy to see the nature of the errors.

| category | meaning |
| -------- | ------- |
| `OK` | the slot was predicted with the correct value |
| `WRONG` | the slot was found, but the value is incorrect |
| `MISSING` | the correct answer contains this slot, but the model did not predict it |
| `SPURIOUS` | the model predicted this slot, but it is not in the correct answer |

## Model Selection

Six models are available. You can select one with the `--model` option.

```bash
uv run wafl_llm_dst_train.py --model qwen3-4b
```

| Alias | Size | Hugging Face id | License |
| ----- | ---- | --------------- | ------- |
| `qwen3-8b` (default) | 8B | `unsloth/Qwen3-8B` | Apache-2.0 |
| `qwen3-4b` | 4B | `unsloth/Qwen3-4B-Instruct-2507` | Apache-2.0 |
| `ministral3-8b` | 8B | `unsloth/Ministral-3-8B-Instruct-2512` | Apache-2.0 |
| `ministral3-3b` | 3B | `unsloth/Ministral-3-3B-Instruct-2512` | Apache-2.0 |
| `llama31-8b` | 8B | `unsloth/Meta-Llama-3.1-8B-Instruct` | Llama 3.1 Community |
| `llama32-3b` | 3B | `unsloth/Llama-3.2-3B-Instruct` | Llama 3.2 Community |

Each family of models uses a different format for writing conversations into a
prompt, and the program handles this difference automatically. Other models
from Hugging Face can also be used. See [docs/models.md](docs/models.md) for
details.

## Baselines and Ablation Studies

An accuracy figure by itself does not prove that the exchange of models is what
produced the improvement. It might be that the devices simply trained for a
long time. To check this, two comparison experiments are provided.

The first experiment disables the exchange entirely, and instead lets every
device train on its own data in every round.

```bash
uv run wafl_llm_dst_train.py --mode self
```

This gives each device several times more training than WAFL does. If WAFL
still produces better results, then the improvement cannot be explained by the
amount of training alone.

The second experiment keeps everything exactly as it is in WAFL, including
which devices train in which round, and only sets the averaging weight to zero
so that nothing is actually exchanged.

```bash
uv run wafl_llm_dst_train.py --fl-coefficiency 0
```

In this case the number of training steps is identical to the WAFL run, so the
only difference in the whole experiment is whether the adapters are combined.
This is the most precise comparison. Both experiments are described in
[docs/experiments.md](docs/experiments.md).

## Experiments with IID Setting

The main subject of this project is the Non-IID setting, because that is what
happens in reality. Each person has different conversations, so each device
holds a different kind of data. For reference, however, the program can also
divide the data in the **IID** way, in which all conversations are shuffled and
then dealt out evenly. Every device then holds roughly the same mixture of the
five domains.

The IID setting is useful as a point of comparison for two reasons. First, it
shows what accuracy is reachable when the uneven distribution of data is not a
problem at all. Second, it tells you how much of the difficulty in the Non-IID
setting comes from the distribution rather than from the task itself.

Add the option `--split-type iid` to use this setting. It can be combined with
`--mode self` in the same way as before.

```bash
# WAFL on IID data
uv run wafl_llm_dst_train.py --split-type iid
uv run wafl_llm_dst_eval.py  --split-type iid

# Self-training only, on IID data
uv run wafl_llm_dst_train.py --mode self --split-type iid
uv run wafl_llm_dst_eval.py  --mode self --split-type iid
```

Together with the Non-IID runs, this gives four experiments. Each one writes to
its own directory, so they never overwrite each other.

| training command | directory | evaluation summary |
| ---------------- | --------- | ------------------ |
| (no options) | `wafl-qwen3-8b-dst-noniid/` | `wafl_qwen3-8b_eval_noniid.json` |
| `--split-type iid` | `wafl-qwen3-8b-dst-iid/` | `wafl_qwen3-8b_eval_iid.json` |
| `--mode self` | `self-qwen3-8b-dst-noniid/` | `self_qwen3-8b_eval_noniid.json` |
| `--mode self --split-type iid` | `self-qwen3-8b-dst-iid/` | `self_qwen3-8b_eval_iid.json` |

The last of these four is worth explaining. When the data is IID and no
exchange takes place, every device is simply learning on its own from a
representative sample of all five domains. This is close to ordinary
centralised training. If WAFL on Non-IID data reaches a similar accuracy, then
the exchange of adapters has recovered almost everything that was lost because
of the uneven distribution.

You can also inspect the individual predictions of an IID run, by giving the
same option to the third program.

```bash
uv run wafl_llm_dst_inspect.py --split-type iid --round 300 --node 9 --only-errors
```

---

## Documentation

| document | contents |
| -------- | -------- |
| [Background](docs/background.md) | why collaborative learning of DST is worth studying |
| [How it works](docs/how-it-works.md) | the algorithm, the averaging formula, and contact patterns |
| [Supported models](docs/models.md) | the model list, prompt formats, and how to add a model |
| [Data split](docs/data-split.md) | how the dataset is divided among the devices |
| [Installation](docs/installation.md) | detailed setup, including how to fix CUDA version problems |
| [Baselines and ablations](docs/experiments.md) | the comparison experiments explained in full |
| [Results](docs/results.md) | all measured results, including an analysis of the errors |
| [Command line reference](docs/cli.md) | every option of every program |
| [Troubleshooting](docs/troubleshooting.md) | errors that have occurred in practice, and their causes |
| [Implementation notes](docs/implementation-notes.md) | design decisions, licensing, and references |

An index of these documents is also available at [docs/README.md](docs/README.md).

## Repository Structure

| file | purpose |
| ---- | ------- |
| `wafl_llm_dst_train.py` | trains the adapters and performs the WAFL exchange |
| `wafl_llm_dst_eval.py` | measures the accuracy of every device |
| `wafl_llm_dst_inspect.py` | shows predictions and correct answers turn by turn |
| `wafl_llm_data_split.py` | divides MultiWOZ 2.4 among the devices |
| `model_registry.py` | model definitions and prompt formats |
| `mwz24_data.py` | downloads the dataset and builds prompts and labels |

## License

This code is distributed under the GNU General Public License v3.0, in the same
way as the rest of this repository. The model weights are separate works and
have their own licenses. In particular, the two Llama models are not covered by
an Apache license. Please see
[docs/implementation-notes.md](docs/implementation-notes.md).
