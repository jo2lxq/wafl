# WAFL-Whisper

This project is a Federated Learning framework for speech recognition using the Whisper model.

## Environment Setup
Please choose either "Singularity image" or "conda"

### Singularity Image
1. Build the Singularity image:

```bash
singularity build --fakeroot wafl_whisper.sif wafl_whisper.def
```

2. Run the container:

```bash
singularity shell --nv --bind /dir/to/WAFL-Whisper:/dir/to/WAFL-Whisper --contain /path/to/wafl_whisper.sif
```

3. Place node movement simulation files here:

Place the necessary files in the contact_file/ directory. Sample RWP-based simulations for 4 and 6 devices are already included.

The sample files should be in JSON format, where each object represents one epoch. Each key is a node ID, and the corresponding value is a list of node IDs it communicates with in that epoch.


### conda

```bash
conda create -n WAFL-Whisper python=3.10 -y
conda activate WAFL-Whisper
pip install -r requirements.txt
```

## Basic Configuration

Before running the project, you need to create a config.json file.
A sample configuration is already provided in the file, and it should include the following parameters:

```json
{
    "memo": "Experiment memo (This memo will be recorded in the output directory where the results are saved)",
    "n_node": "Number of nodes",
    "pre_epoch": "Number of pre-training epochs",
    "num_epoch": "Number of collaborative learning epochs",
    "lr": "Learning rate",
    "contact_file": "Path to communication pattern file",
    "data_dir": "Path to data directory",
    "data_name_list": ["List of folders to use in data_dir"],
    "fl_coefficiency": "Coefficient for model parameter synthesis (between 0 and 1)",
    "seed": "Random seed",
    "output_dir": "Path to output directory",
    "train_batch_size": "Training batch size",
    "test_batch_size": "Testing batch size"
}
```

## Dataset Preparation

1. Data directory structure:

```
data_dir/
    ├── dataset1/              # Dataset for each node
    │   ├── audio/             # Store audio files
    │   │   ├── sample1.wav    # Audio file
    │   │   ├── sample2.wav
    │   │   └── ...
    │   └── script/            # Store text files
    │       ├── sample1.txt    # Text corresponding to audio file
    │       ├── sample2.txt
    │       └── ...
    ├── dataset2/              # Dataset for another node
    │   ├── audio/
    │   └── script/
    └── test/                  # Test data
        ├── audio/
        └── script/
```

2. Dataset requirements:

- Each dataset folder (dataset1, dataset2, test) must have two subfolders: `audio` and `script`
- The `audio` folder contains audio files (.wav)
- The `script` folder contains corresponding text files (.txt)
- Audio files and text files are matched by having the same name (with different extensions)
  - Example: `audio/sample1.wav` ↔ `script/sample1.txt`

3. Sample Dataset
- You can download a sample dataset from [here](https://drive.google.com/file/d/1ZvJO4Argwu6ZK0Rwacd_mTj2iw6yrMb6/view).
- Place the dataset directly under WAFL-Whisper so you can run the code without creating your own dataset.  
- The dataset is derived from [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0). Because some files have poor audio quality and each node contains audio from multiple speakers, training may not work properly and the CER may not improve significantly.


## Execution

Run the following command:

```bash
python main.py
```

## Output

After execution, results will be saved in the following directory structure:

```
output_dir/
    ├── graph/              # Graph images
    │   └── average_cer.png # Graph of average CER progression
    ├── text/               # Text output
    │   └── node{0..n}/     # Output directories containing transcription results for test data
    ├── model/              # Trained models
    ├── all_result.txt      # Summary of experiment settings and results
    └── cer_results.json    # CER results (JSON format)
```

### Output File Description

1. `all_result.txt`

   - Experiment settings (contents of config.json)
   - Execution date and time
   - Final average CER

2. `cer_results.json`
   - CER for each node across all epochs and average CER in JSON format
   - Structure example:
     ```json
     {
       "node_results": {
         "node_0": [0.123, 0.115, 0.108, ...],  // CER for node 0 at each epoch
         "node_1": [0.134, 0.128, 0.121, ...],  // CER for node 1 at each epoch
         ...
       },
       "average_results": [0.129, 0.122, 0.115, ...]  // Progression of average CER across all nodes
     }
