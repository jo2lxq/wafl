# WAFL Emulation Testbed

The WAFL Emulation Testbed project, launched in April 2025, is trying to develop a research platform for WAFL's device-to-device collaborative learning over real TCP/IP-based communications. The previous researches were mainly conducted by "simulation", which executes "model exchange" on the memory space in a single computer. This emulation platform has 10 devices (assuming 10 GPU-equipped computers) and one control server connected in a local area network. Even though these devices can always be connected over the Ethernet, device-to-device contact patterns can be defined and controlled to ensure the reproducibility of the experiments.

<img src="./assets/wafl-emulation-testbed.png">

## Setup

1. Install [mise](https://mise.jdx.dev/getting-started.html).
1. Run the command below to install python and dependencies:

```bash
mise setup
```

## Development

When you try to commit, `pre-commit` will automatically format the code and check for linting errors.
If you want to run it manually, you can use:

```bash
mise lint
```

