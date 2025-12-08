# EvOlf: Source Code & Development Repository

Welcome to the source code repository for **EvOlf**, a deep-learning framework for predicting ligand-GPCR interactions.

> **Note for Users:** If you simply want to *run* the pipeline on your data, please visit the main pipeline repository:  
>    **[the-ahuja-lab/evolf-pipeline](https://github.com/the-ahuja-lab/evolf-pipeline)**
>
> This repository (`evolf-pipeline-source`) contains the raw scripts, Dockerfiles, environment configurations, and module definitions used to *build* and *maintain* the pipeline. It is intended for developers, contributors, and advanced users who wish to modify the underlying logic.

## Repository Structure

This repository is organized to separate the Nextflow orchestration logic from the scientific computation scripts and their environments.

```bash
.
├── main.nf                 # The main Nextflow orchestration script
├── nextflow.config         # Global configuration (Profiles, Docker, GPU settings)
│
├── modules/                # Nextflow Process Definitions (The "Wrappers")
│   └── local/
│       ├── prepare_input/  # Input standardization process
│       ├── chemberta/      # Wrapper for ChemBERTa featurization
│       ├── protbert/       # Wrapper for ProtBERT featurization
│       └── ... (wrappers for all 12+ processes)
│
└── envs/                   # The Core Logic & Environments (The "Guts")
    ├── EvOlf_DL/       # Shared Deep Learning Environment
    │   ├── Dockerfile      # Builds the 'evolfdl_env' image
    │   ├── env_gpu.lock.yml# Conda lock file for reproducibility
    │   ├── ChemBERTa.py    # Python script for ligand embedding
    │   ├── ProtBERT.py     # Python script for receptor embedding
    │   └── ProtT5.py       # Python script for receptor embedding
    │
    ├── signaturizer/       # Bioactivity Signature Environment
    │   ├── Dockerfile
    │   ├── Signaturizer.py
    │   └── ...
    │
    └── ... (folders for EvOlf_R, mol2vec, graph2vec, etc.)
```

## Development & Customization

### 1\. Modifying Scientific Logic (Python/R)

The actual scientific computation happens in the scripts located in `envs/`.

  * To change how ChemBERTa embeddings are generated, edit `envs/EvOlf_DL/ChemBERTa.py`.
  * To modify the final prediction model architecture, edit `envs/evolf_prediction/EvOlfPrediction.py`.

### 2\. Updating Environments (Docker/Conda)

We use a "sealed bubble" approach for reproducibility.

  * **Environments** are defined in `envs/<module>/<name>.yml`.
  * **Lock files** (`.yml`) are used in Dockerfiles to guarantee exact package versions.
  * **Dockerfiles** are located in each subdirectory of `envs/`.

**To update an environment:**

1.  Create recipe YAML file (e.g., `envs/signaturizer/Signaturizer.yml`).
2.  Re-generate the lock file using Conda.
3.  Re-build the Docker image:
    ```bash
    docker build -f envs/signaturizer/Dockerfile -t ahujalab/signaturizer_env:latest .
    ```
4.  Push the new image to the registry (required for the main pipeline to see changes).

### 3\. Modifying the Pipeline Flow (Nextflow)

The orchestration logic resides in `modules/` and `main.nf`.

  * **`modules/local/<process>/main.nf`**: Defines the input/output channels and the command string for a single step.
  * **`main.nf`**: Connects these modules into the final workflow (e.g., parallel fan-out, synchronization, fan-in).

## Container Images

This pipeline relies on pre-built images hosted on Docker Hub. The Dockerfiles in this repository correspond to these images:

| Repository | Description | Source Path |
| :--- | :--- | :--- |
| `ahujalab/evolfdl_env` | Shared DL environment (PyTorch, Transformers) | `envs/EvOlf_DL/` |
| `ahujalab/evolfprediction_env` | Prediction model environment | `envs/evolf_prediction/` |
| `ahujalab/evolfr_env` | R environment for data prep & compilation | `envs/EvOlf_R/` |
| `ahujalab/signaturizer_env` | TensorFlow env for Signaturizer | `envs/signaturizer/` |
| ... | (See `envs/` for full list) | ... |

## Running the Pipeline

### Installation & Prerequisites

You do not need to install Python libraries or R packages manually. The only requirements are Nextflow and a container engine.

1.  **Install [Nextflow](https://www.nextflow.io/)** (Requires Java 11+):

    ```bash
    conda create -n nf-env bioconda::nextflow
    conda activate nf-env
    ```

2.  **Install a Container Engine:**

      * [**Docker Desktop / Engine:**](https://docker.com/) Best for local development and workstations.
      * **[Apptainer](https://apptainer.org/) (formerly [Singularity](https://sylabs.io/singularity/)):** **Required** for HPC clusters and shared servers.

-----

### System Requirements

1. EvOlf pipeline requires a minimum of 12 GB to run properly as it utilises various PyTorch and TensorFlow models for generating featurizers.
2. To make use of parallel processing provided by Nextflow, a multi-core processor system is required.
3. Although a GPU is not required to run EvOlf, it is still a preferable choice to run EvOlf on a GPU with a minimum of 12GB vRAM to make the pipeline quick.

Kindly refer to [Hardware Configuration](#hardware-profiles--profile) section to see how to configure hardware settings and profiles as per your system.

-----

### Input Format

The pipeline expects a **CSV file** with at least two columns: the Ligand SMILES and the Receptor Amino Acid Sequence.

#### Recommended Format (With IDs)

Providing your own IDs is the safest way to track your results.

```csv
ID,Ligand_ID,SMILES,Receptor_ID,Receptor_Sequence
Pair_1,Lig_A,CCO,Rec_1,MGA...
Pair_2,Lig_B,CCN,Rec_1,MGA...
```

#### Minimal Format (No IDs)

EvOlf can auto-generate IDs for you.

```csv
SMILES,Sequence
CCO,MGA...
CCN,MGA...
```

> **⚠️ Crucial Tip:** Convert your SMILES to **Canonical Format** before running the pipeline\!
>
> ```bash
> obabel -ismi input.smi -ocan -O canonical.smi
> ```
>
> Non-canonical SMILES can lead to inconsistent feature generation.

-----

### Usage

#### 1\. Single-File Mode

Use this for running a single dataset.

```bash
nextflow run the-ahuja-lab/evolf-pipeline \
 --inputFile "data/my_experiment.csv" \
    --ligandSmiles "SMILES" \
 --receptorSequence "Sequence" \
    --outdir "./results/experiment_1" \
 -profile docker,gpu
```

#### 2\. Batch Mode (High-Throughput)

Use this to process multiple datasets in parallel. Create a manifest CSV file:

**`batch_manifest.csv`**:

```csv
inputFile,ligandSmiles,receptorSequence,ligandID,receptorID,lrID
/data/project_A.csv,SMILES,Seq,L_ID,R_ID,Pair_ID
/data/project_B.csv,cano_smiles,aa_seq,lig_name,prot_name,int_id
```

**Run Command:**

```bash
nextflow run the-ahuja-lab/evolf-pipeline \
 --batchFile "batch_manifest.csv" \
    --outdir "./results/batch_run" \
 -profile docker,gpu
```

-----

### Output Explanation

The pipeline creates a separate folder for each input file in your output directory.

```
results/
├── my_experiment/
│   ├── my_experiment_Prediction_Output.csv    <-- FINAL RESULTS
│   ├── my_experiment_Input_ID_Information.csv <-- ID Mapping Key
│   └── pipeline_info/                           
│       ├── execution_report.html              <-- Resource usage (RAM/CPU)
│       └── execution_timeline.html            <-- Gantt chart of jobs
```

#### The Prediction File (`Prediction_Output.csv`)

| Column | Description |
| :--- | :--- |
| `ID` | The unique Pair ID (User-provided or auto-generated `LR1`). |
| `Predicted Label` | **1** (Interaction) or **0** (No Interaction). Threshold is 0.5. |
| `P1` | The raw **probability score** (0.0 to 1.0). Higher = stronger interaction confidence. |

-----

### Configuration & Tuning

#### Hardware Profiles (`-profile`)

  * **`docker`**: Uses Docker. Runs as the current user (`-u $(id -u)`) to prevent root-owned file issues.
  * **`apptainer` / `singularity`**: Uses `.sif` images. Automatically mounts your project directory.
  * **`gpu`**: **Highly Recommended.** Enables CUDA for ChemBERTa, ProtBERT, ProtT5, and the Prediction model. Without this, the pipeline will run on CPU (which is much slower).

#### Model Caching

The first time you run EvOlf, it will download \~15GB of model weights (Hugging Face transformers).

  * **Location:** These are stored in `~/.evolf_cache` in your home directory.
  * **Benefit:** They are downloaded **only once**. All subsequent runs (even in different project folders) will use this centralised cache.

#### Resource Management

The Nextflow pipeline utilises 4 CPU cores and 16GB of RAM by default. You can change these defaults by providing the following parameters along with input:
- `--maxCPUs` to set the maximum number of CPU cores NextFlow can utilise to run EvOlf.
- `--maxMemory` to set the maximum amount of RAM the EvOlf pipeline can use.

-----

### Troubleshooting

**Q: `Permission denied: '/.cache'`**

  * **Fix:** This usually happens if the cache directory isn't set correctly. Ensure you haven't manually modified the `modules/` scripts. The pipeline is configured to use `~/.evolf_cache` automatically.

**Q: `CUDA out of memory`**

  * **Fix:** Deep learning models are memory-hungry. The pipeline automatically limits GPU jobs (`maxForks = 1`) to prevent crashing, but if you still see this, try reducing the batch size in your input file or running on a GPU with more VRAM (16GB+ recommended).

**Q: `docker: invalid reference format`**

  * **Fix:** You are likely using an older version of Nextflow or Docker. Ensure you are using `-profile docker` and not trying to build the image yourself. The pipeline pulls pre-built images from Docker Hub.

-----

## Credits & Citations

The EvOlf Pipeline was developed by the **Ahuja Lab** at IIIT-Delhi.

### Principal Investigator

  * **Dr. Gaurav Ahuja** ([@the-ahuja-lab](https://github.com/the-ahuja-lab))

### Lead Developers

  * **Adnan Raza** ([@woosflex](https://github.com/woosflex))
  * **Syed Yasser** ([@yasservision24](https://github.com/yasservision24))
  * **Pranjal Sharma** ([@PRANJAL2208](https://github.com/PRANJAL2208))
  * **Saveena Solanki** ([@SaveenaSolanki](https://github.com/SaveenaSolanki))
  * **Aayushi Mittal** ([@Aayushi006](https://github.com/Aayushi006))
  * **Mudit Gupta** ([@MaddyG-05](https://github.com/MaddyG-05))

### Related Repositories

  * **Bare Minimum Pipeline:** [evolf-pipeline](https://github.com/the-ahuja-lab/evolf-pipeline)
  * **Source Code, Development & Training Code (This Repo):** [EvOlf](https://www.google.com/search?q=https://github.com/the-ahuja-lab/EvOlf)

-----