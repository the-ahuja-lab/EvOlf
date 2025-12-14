# EvOlf Nextflow Pipeline for Local Run

## Quick Start

(Assuming you docker or any container engine installed)

1. Initialize the environment
  ```
  conda create -n evolf-env bioconda::nextflow
  ```
2. Prepare Input File using the prescribed structure
  ```
  SMILES,Sequence
  CCO,MGA...
  CCN,MGA...
  ```
3. Run EvOlf Pipeline
  ```
  nextflow run the-ahuja-lab/evolf-pipeline --inputFile my_lr_pair.csv --ligandSmiles SMILES --receptorSequence Sequence -profile docker
  ```

Refer to section [Running the Pipeline](#running-the-pipeline) for more detailed commands and options.

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

## Pipeline Architecture

The EvOlf pipeline is a modular, multi-stage workflow built on Nextflow. It parallelises the heavy lifting of feature generation across multiple containers.

### 1\. Input Standardisation (`PREPARE_INPUT`)

  - Validates input CSV formats.
  - Standardises SMILES strings and checks for invalid characters.
  - Maps user-provided IDs or generates auto-incrementing IDs (`L1`, `R1`, `LR1`) to track interactions throughout the pipeline.

### 2\. Multi-Modal Featurization

EvOlf uses a "consensus" approach, combining multiple representations of both ligands and receptors to capture chemical and biological nuances.

**Ligand Featurizers (Small Molecules):**

  * **ChemBERTa:** A transformer model pre-trained on 77M SMILES strings for learning molecular substructures.
  * **Signaturizer:** Infers bioactivity signatures (MoA) based on the Chemical Checker.
  * **Mol2Vec:** An unsupervised method (Word2Vec-based) to learn vector representations of molecular substructures.
  * **Mordred:** Calculates over 1,600 2D and 3D physicochemical descriptors.
  * **Graph2Vec:** Represents molecules as graphs to capture topological structures.

**Receptor Featurizers (Proteins):**

  * **ProtBERT:** A massive protein language model (BERT-based) trained on UniRef100.
  * **ProtT5:** A T5-based encoder for capturing long-range dependencies in protein sequences.
  * **ProtR:** Calculates physicochemical properties (hydrophobicity, charge, etc.) from amino acid sequences.
  * **MathFeaturizer:** A custom module encoding sequences using mathematical descriptors (chaos game representation, etc.).

### 3\. Deep Learning Inference (`EVOLF_PREDICTION`)

  - **Feature Compilation:** Aggregates the diverse feature sets into a unified tensor.
  - **Prediction:** Passes the tensors through the trained EvOlf deep neural network.
  - **Output:** Returns a probability score (`0.0` to `1.0`) indicating the likelihood of interaction.

---

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

1. EvOlf pipeline requires a minimum of 12 GB memory to run properly as it utilises various PyTorch and TensorFlow models for generating featurizers.
2. To make use of parallel processing provided by Nextflow, a multi-core processor system is required.
3. Although a GPU is not required to run EvOlf, it is still a preferable choice to run EvOlf on a GPU with a minimum of 12GB vRAM to make the pipeline quick.

Kindly refer to [Hardware Configuration](#hardware-profiles--profile) section to see how to configure hardware settings and profiles as per your system.

-----
### Input Format

Before you unleash EvOlf’s predictive power, make sure your input data is formatted correctly! Each row should represent a single ligand-receptor interaction and must include:  

1. **SMILES** – The molecular representation of the ligand.    
    - **Important Note on SMILES Format** <br> 
 For optimal prediction accuracy, we strongly recommend converting your ligand SMILES to canonical SMILES format using OpenBabel before running EvOlf. This standardisation ensures consistency in molecular representation and significantly improves model performance.
      ```bash
      # Example of converting to canonical SMILES using OpenBabel
      obabel -ismi input.smi -ocan -O out.smi
      ```
2. **Receptor Sequence** – The amino acid sequence of the GPCR.

To keep things organised, you can provide your own **unique identifiers** for ligands, receptors, and ligand-receptor pairs, or if you prefer, EvOlf can generate them for you automatically. 
Let’s go through both options!

---
<br> 

#### Simplest Format: No IDs Required!
If manually assigning identifiers sounds like a hassle, you can skip them entirely! Just provide a CSV file with only ligand SMILES and receptor sequences, and EvOlf will automatically generate the necessary IDs for you.

🔹 No IDs Provided Example
| SMILES | Sequence |
|:-----------|:-----------|
|OCC/C=C\CC  | MEPRKNVTDFVLLGFTQNPKEQKVLFVMFLLFYILTMVGNLLIVVTVTVSETLGSPMSFFLAGLTFIDIIYSSSISPRLISDLFFGNNSISFQSFMAQLFIEHLFGGSEVFLLLVMAYDRYVAICKPLHYLVIMRQWVCVLLLVVSWVGGFLQSVFQLSIIYGLPFCGPNVIDHFFCDMYPLLKLACTDTHVIGLLVVANGGLSCTIAFLLLLISYGVILHSLKKLSQKGRQKAHSTCSSHITVVVFFFVPCIFMCARPARTFSIDKSVSVFYTVITPMLNPLIYTLRNSEMTSAMKKL|
|OCC/C=C\CC  | MIPIQLTVFFMIIYVLESLTIIVQSSLIVAVLGREWLQVRRLMPVDMILISLGISRFCLQWASMLNNFCSYFNLNYVLCNLTITWEFFNILTFWLNSLLTVFYCIKVSSFTHHIFLWLRWRILRLFPWILLGSLMITCVTIIPSAIGNYIQIQLLTMEHLPRNSTVTDKLENFHQYQFQAHTVALVIPFILFLASTIFLMASLTKQIQHHSTGHCNPSMKARFTALRSLAVLFIVFTSYFLTILITIIGTLFDKRCWLWVWEAFVYAFILMHSTSLMLSSPTLKRILKGKC|
|CCCC(=O)O  | MGRGNSTEVTEFHLLGFGVQHEFQHVLFIVLLLIYVTSLIGNIGMILLIKTDSRLQTPMYFFPQHLAFVDICYTSAITPKMLQSFTEENNLITFRGCVIQFLVYATFATSDCYLLAIMAMDCYVAICKPLRYPMIMSQTVYIQLVAGSYIIGSINASVHTGFTFSLSFCKSNKINHFFCDGLPILALSCSNIDINIILDVVFVGFDLMFTELVIIFSYIYIMVTILKMSSTAGRKKSFSTCASHLTAVTIFYGTLSYMYLQPQSNNSQENMKVASIFYGTVIPMLNPLIYSLRNKEGK|
|CCCCCCCC=O  | MGRGNSTEVTEFHLLGFGVQHEFQHVLFIVLLLIYVTSLIGNIGMILLIKTDSRLQTPMYFFPQHLAFVDICYTSAITPKMLQSFTEENNLITFRGCVIQFLVYATFATSDCYLLAIMAMDCYVAICKPLRYPMIMSQTVYIQLVAGSYIIGSINASVHTGFTFSLSFCKSNKINHFFCDGLPILALSCSNIDINIILDVVFVGFDLMFTELVIIFSYIYIMVTILKMSSTAGRKKSFSTCASHLTAVTIFYGTLSYMYLQPQSNNSQENMKVASIFYGTVIPMLNPLIYSLRNKEGK|
|CC(=O)C(=O)C  | MVGANHSVVSEFVFLGLTNSWEIRLLLLVFSSMFYMASMMGNSLILLTVTSDPHLHSPMYFLLANLSFIDLGVSSVTSPKMIYDLFRKHEVISFGGCIAQIFFIHVIGGVEMVLLIAMAFDRYVAICKPLQYLTIMSPRMCMFFLVAAWVTGLIHSVVQLVFVVNLPFCGPNVSDSFYCDLPRFIKLACTDSYRLEFMVTANSGFISLGSFFILIISYVVIILTVLKHSSAGLSKALSTLSAHVSVVVLFFGPLIFVYTWPSPSTHLDKFLAIFDAVLTPVLNPIIYTFRN|

When using this format, EvOlf will generate a file called `Input_ID_Information.csv` in the output directory, which contains automatically assigned Ligand IDs, Receptor IDs, and Ligand-Receptor Pair IDs.

---
<br> 

#### Provide Unique Identifiers
If you prefer more control over your data, you can provide custom, unique IDs for ligands, receptors, and ligand-receptor pairs. This helps keep your dataset structured and makes it easier to track specific interactions.

✅ Good Input Data Example
| ID | Ligand_ID | SMILES | Receptor_ID| Sequence |
|:-----|:---------|:-----------|:---------|:-----------|
|LR1  |L1  |OCC/C=C\CC  |R1  | MEPRKNVTDFVLLGFTQNPKEQKVLFVMFLLFYILTMVGNLLIVVTVTVSETLGSPMSFFLAGLTFIDIIYSSSISPRLISDLFFGNNSISFQSFMAQLFIEHLFGGSEVFLLLVMAYDRYVAICKPLHYLVIMRQWVCVLLLVVSWVGGFLQSVFQLSIIYGLPFCGPNVIDHFFCDMYPLLKLACTDTHVIGLLVVANGGLSCTIAFLLLLISYGVILHSLKKLSQKGRQKAHSTCSSHITVVVFFFVPCIFMCARPARTFSIDKSVSVFYTVITPMLNPLIYTLRNSEMTSAMKKL|
|LR2  |L1  |OCC/C=C\CC  |R2  | MIPIQLTVFFMIIYVLESLTIIVQSSLIVAVLGREWLQVRRLMPVDMILISLGISRFCLQWASMLNNFCSYFNLNYVLCNLTITWEFFNILTFWLNSLLTVFYCIKVSSFTHHIFLWLRWRILRLFPWILLGSLMITCVTIIPSAIGNYIQIQLLTMEHLPRNSTVTDKLENFHQYQFQAHTVALVIPFILFLASTIFLMASLTKQIQHHSTGHCNPSMKARFTALRSLAVLFIVFTSYFLTILITIIGTLFDKRCWLWVWEAFVYAFILMHSTSLMLSSPTLKRILKGKC|
|LR3  |L2  |CCCC(=O)O  |R3  | MGRGNSTEVTEFHLLGFGVQHEFQHVLFIVLLLIYVTSLIGNIGMILLIKTDSRLQTPMYFFPQHLAFVDICYTSAITPKMLQSFTEENNLITFRGCVIQFLVYATFATSDCYLLAIMAMDCYVAICKPLRYPMIMSQTVYIQLVAGSYIIGSINASVHTGFTFSLSFCKSNKINHFFCDGLPILALSCSNIDINIILDVVFVGFDLMFTELVIIFSYIYIMVTILKMSSTAGRKKSFSTCASHLTAVTIFYGTLSYMYLQPQSNNSQENMKVASIFYGTVIPMLNPLIYSLRNKEGK|
|LR4  |L3  |CCCCCCCC=O  |R3  | MGRGNSTEVTEFHLLGFGVQHEFQHVLFIVLLLIYVTSLIGNIGMILLIKTDSRLQTPMYFFPQHLAFVDICYTSAITPKMLQSFTEENNLITFRGCVIQFLVYATFATSDCYLLAIMAMDCYVAICKPLRYPMIMSQTVYIQLVAGSYIIGSINASVHTGFTFSLSFCKSNKINHFFCDGLPILALSCSNIDINIILDVVFVGFDLMFTELVIIFSYIYIMVTILKMSSTAGRKKSFSTCASHLTAVTIFYGTLSYMYLQPQSNNSQENMKVASIFYGTVIPMLNPLIYSLRNKEGK|
|LR5  |L4  |CC(=O)C(=O)C  |R4  | MVGANHSVVSEFVFLGLTNSWEIRLLLLVFSSMFYMASMMGNSLILLTVTSDPHLHSPMYFLLANLSFIDLGVSSVTSPKMIYDLFRKHEVISFGGCIAQIFFIHVIGGVEMVLLIAMAFDRYVAICKPLQYLTIMSPRMCMFFLVAAWVTGLIHSVVQLVFVVNLPFCGPNVSDSFYCDLPRFIKLACTDSYRLEFMVTANSGFISLGSFFILIISYVVIILTVLKHSSAGLSKALSTLSAHVSVVVLFFGPLIFVYTWPSPSTHLDKFLAIFDAVLTPVLNPIIYTFRN|

This is exactly how your data should be structured:

✅ Each unique SMILES has a corresponding unique Ligand ID. <br> 
✅ Each unique receptor sequence has a corresponding unique Receptor ID. <br> 
✅ Each ligand-receptor pair (row) has a unique ID. <br> 

This structure ensures EvOlf correctly maps interactions without confusion or redundancy.

---
<br>

#### Common Mistakes to Avoid <br>
Providing incorrect or inconsistent identifiers can cause errors in prediction. Below is an example of what NOT to do:

Bad Input Data Example
| ID | Ligand_ID | SMILES | Receptor_ID| Sequence |
|:-----|:---------|:-----------|:---------|:-----------|
|LR1  |L1  |OCC/C=C\CC  |R1  | MEPRKNVTDFVLLGFTQNPKEQKVLFVMFLLFYILTMVGNLLIVVTVTVSETLGSPMSFFLAGLTFIDIIYSSSISPRLISDLFFGNNSISFQSFMAQLFIEHLFGGSEVFLLLVMAYDRYVAICKPLHYLVIMRQWVCVLLLVVSWVGGFLQSVFQLSIIYGLPFCGPNVIDHFFCDMYPLLKLACTDTHVIGLLVVANGGLSCTIAFLLLLISYGVILHSLKKLSQKGRQKAHSTCSSHITVVVFFFVPCIFMCARPARTFSIDKSVSVFYTVITPMLNPLIYTLRNSEMTSAMKKL|
|LR2  |L2  |OCC/C=C\CC  |R1  | MIPIQLTVFFMIIYVLESLTIIVQSSLIVAVLGREWLQVRRLMPVDMILISLGISRFCLQWASMLNNFCSYFNLNYVLCNLTITWEFFNILTFWLNSLLTVFYCIKVSSFTHHIFLWLRWRILRLFPWILLGSLMITCVTIIPSAIGNYIQIQLLTMEHLPRNSTVTDKLENFHQYQFQAHTVALVIPFILFLASTIFLMASLTKQIQHHSTGHCNPSMKARFTALRSLAVLFIVFTSYFLTILITIIGTLFDKRCWLWVWEAFVYAFILMHSTSLMLSSPTLKRILKGKC|
|LR3  |L3  |CCCC(=O)O  |R2  | MGRGNSTEVTEFHLLGFGVQHEFQHVLFIVLLLIYVTSLIGNIGMILLIKTDSRLQTPMYFFPQHLAFVDICYTSAITPKMLQSFTEENNLITFRGCVIQFLVYATFATSDCYLLAIMAMDCYVAICKPLRYPMIMSQTVYIQLVAGSYIIGSINASVHTGFTFSLSFCKSNKINHFFCDGLPILALSCSNIDINIILDVVFVGFDLMFTELVIIFSYIYIMVTILKMSSTAGRKKSFSTCASHLTAVTIFYGTLSYMYLQPQSNNSQENMKVASIFYGTVIPMLNPLIYSLRNKEGK|
|LR4  |L3  |CCCCCCCC=O  |R3  | MGRGNSTEVTEFHLLGFGVQHEFQHVLFIVLLLIYVTSLIGNIGMILLIKTDSRLQTPMYFFPQHLAFVDICYTSAITPKMLQSFTEENNLITFRGCVIQFLVYATFATSDCYLLAIMAMDCYVAICKPLRYPMIMSQTVYIQLVAGSYIIGSINASVHTGFTFSLSFCKSNKINHFFCDGLPILALSCSNIDINIILDVVFVGFDLMFTELVIIFSYIYIMVTILKMSSTAGRKKSFSTCASHLTAVTIFYGTLSYMYLQPQSNNSQENMKVASIFYGTVIPMLNPLIYSLRNKEGK|
|LR4  |L4  |CC(=O)C(=O)C  |R4  | MVGANHSVVSEFVFLGLTNSWEIRLLLLVFSSMFYMASMMGNSLILLTVTSDPHLHSPMYFLLANLSFIDLGVSSVTSPKMIYDLFRKHEVISFGGCIAQIFFIHVIGGVEMVLLIAMAFDRYVAICKPLQYLTIMSPRMCMFFLVAAWVTGLIHSVVQLVFVVNLPFCGPNVSDSFYCDLPRFIKLACTDSYRLEFMVTANSGFISLGSFFILIISYVVIILTVLKHSSAGLSKALSTLSAHVSVVVLFFGPLIFVYTWPSPSTHLDKFLAIFDAVLTPVLNPIIYTFRN|

**What's Wrong Here?** <br>
- **Repeated Ligand IDs for Different SMILES** → "L3" is assigned to two different SMILES structures. Each ligand should have one unique ID. <br>
- **Different Ligand IDs for the Same SMILES** → "L1" and "L2" both have the same SMILES (OCC/C=C\CC). A single ligand should always have a consistent Ligand_ID. <br> 
- **Repeated Receptor IDs for Different Sequences** → "R1" is assigned to two different receptor sequences. Each receptor sequence should have one unique ID. <br> 
- **Different Receptor IDs for the Same Sequence** → "R2" and "R3" have the exact same receptor sequence, yet they are assigned different Receptor_IDs. Each unique receptor sequence should have one consistent Receptor_ID to avoid confusion. <br>
- **Duplicate Ligand-Receptor Pair IDs** → "LR4" appears twice. Every row should have a unique ID for proper tracking. <br> 

Data like this will break EvOlf’s ability to correctly associate ligands and receptors!

---
<br>

**Let EvOlf Handle the IDs!** <br>
If you don't want to manually assign identifiers. Just provide a CSV with ligand SMILES and receptor sequences, and EvOlf will take care of the rest. The assigned IDs will be available in the `Input_ID_Information.csv` file in the output directory.

---
<br>

#### Key Takeaways
✔ Option 1: Provide only SMILES and Receptor Sequences, and let EvOlf handle the rest. <br> 
✔ Option 2: Provide custom unique IDs for ligands, receptors, and ligand-receptor pairs for better tracking. <br> 
✔ Make sure all IDs are unique to avoid errors in predictions. <br> 
✔ If you skip IDs, check `Input_ID_Information.csv` for automatically assigned identifiers. <br> 

Now you’re all set to get your predictions! <br> <br>

-----

## Usage

### 1\. Single-File Mode

Use this for running a single dataset.

```bash
nextflow run the-ahuja-lab/evolf-pipeline \
 --inputFile "data/my_experiment.csv" \
    --ligandSmiles "SMILES" \
 --receptorSequence "Sequence" \
    --outdir "./results/experiment_1" \
 -profile docker,gpu
```

### 2\. Batch Mode (High-Throughput)

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

---

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
### The Prediction File (`Prediction_Output.csv`)

| Column | Description |
| :--- | :--- |
| `ID` | The unique Pair ID (User-provided or auto-generated `LR1`). |
| `Predicted Label` | **1** (Interaction) or **0** (No Interaction). Threshold is 0.5. |
| `P1` | The raw **probability score** (0.0 to 1.0). Higher = stronger interaction confidence. |

-----

## Configuration & Tuning

### Hardware Profiles (`-profile`)

  * **`docker`**: Uses Docker. Runs as the current user (`-u $(id -u)`) to prevent root-owned file issues.
  * **`apptainer` / `singularity`**: Uses `.sif` images. Automatically mounts your project directory.
  * **`gpu`**: **Highly Recommended.** Enables CUDA for ChemBERTa, ProtBERT, ProtT5, and the Prediction model. Without this, the pipeline will run on CPU (which is much slower).

### Model Caching

The first time you run EvOlf, it will download \~15GB of model weights (Hugging Face transformers).

  * **Location:** These are stored in `~/.evolf_cache` in your home directory.
  * **Benefit:** They are downloaded **only once**. All subsequent runs (even in different project folders) will use this centralised cache.

### Resource Management

The Nextflow pipeline utilises 4 CPU cores and 16GB of RAM by default. You can change these defaults by providing the following parameters along with input:
- `--maxCPUs` to set the maximum number of CPU cores NextFlow can utilise to run EvOlf.
- `--maxMemory` to set the maximum amount of RAM the EvOlf pipeline can use.

-----

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

### Related Links

  * **EvOlf Webserver:** [http://evolf.ahujalab.iiitd.edu.in/](http://evolf.ahujalab.iiitd.edu.in/)
  * **Docker Images:** All the container images used by EvOlf Pipeline are hosted on DockerHub at [ahujalab](https://hub.docker.com/u/ahujalab).
  * **Source Code, Development & Training Code (This Repo):** [EvOlf](https://www.google.com/search?q=https://github.com/the-ahuja-lab/EvOlf)

-----