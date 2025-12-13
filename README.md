# EvOlf: Evolutionary-Guided Advanced Deep Learning Architecture Powers Mammalian GPCRome Agonist Predictions
<div align="center">
<img src="EvOlf.png", width=600></div>
<br>

<!--- BADGES --->
[![Web Server](https://img.shields.io/badge/EvOlf-Web_Server-red?style=flat&labelColor=black)](https://evolf.ahujalab.iiitd.edu.in/)
[![GitHub](https://img.shields.io/badge/GitHub-EvOlf_Source_Code-blue?style=flat&labelColor=black&logo=github&logoColor=#181717)](https://github.com/the-ahuja-lab/EvOlf/)
[![GitHub](https://img.shields.io/badge/GitHub-EvOlf_Pipeline-green?style=flat&labelColor=black&logo=github&logoColor=#181717)](https://github.com/the-ahuja-lab/evolf-pipeline/)
<!--- Update colab path --->
<!--- ADD Paper when published --->

EvOlf is a cutting-edge deep-learning framework designed to predict ligand-GPCR interactions, integrating odorant and non-odorant GPCRs across 20+ species. Using state-of-the-art modelling techniques enables accurate predictions and GPCR deorphanization. <br><br>
Explore our work through our [EvOlf WebServer](https://evolf.ahujalab.iiitd.edu.in/), where you can browse the complete EvOlf dataset of ligand-receptor interactions used in training and validation. <br><br>
Want to see EvOlf in action? We’ve made it easy for you to run predictions on your own data. You have two ways to get started:

1️⃣ **Try the EvOlf Model** - If you want to check how EvOlf gives output and validate the model, you can use our web-hosted [Evolf Prediction Model](https://evolf.ahujalab.iiitd.edu.in/prediction). 

> [!Important]
> As we are hosting EvOlf on a shared academic server, we currently limit the model to only predict a single ligand-GPCR interaction at a time. If you have multiple ligand-GPCR interactions to predict, kindly use the methods below.
<br> 

2️⃣ **Run using Nextflow (Easiest Option!)** - If you want to run EvOlf locally and without restriction, follow the instructions in the [Get Started](#get-started) section. After initial setup, you'll be able to run the EvOlf model - quick and hassle-free! <br> 

👉 [Run on Server](https://evolf.ahujalab.iiitd.edu.in/prediction) | [Run using Nextflow](https://github.com/the-ahuja-lab/evolf-pipeline) | [Look at the source code](https://github.com/the-ahuja-lab/evolf-pipeline-source)

<br>

## Get Started
### Prerequisites
Ready to dive in? 
1. First, set up a container engine of your liking on your system. Two of the most popular options are as follows:
   - **Docker**: If you want to run the EvOlf pipeline on a local machine or a dedicated workstation with root access, install and set up [Docker Engine](https://docs.docker.com/engine/install/) (GNU-Linux-based OS) or [Docker Desktop](https://docs.docker.com/desktop/) (Windows/MacOS). If you don't have root access, which is common when using shared workstations, you can set up or ask the IT team to allow [rootless Docker](https://docs.docker.com/engine/security/rootless/).
   - **Apptainer**: EvOlf pipeline is compatible with [Apptainer](https://apptainer.org/) (formerly [Singularity](https://docs.sylabs.io/guides/3.5/user-guide/index.html)) to allow EvOlf prediction model to run HPCs.
2. Install [Nextflow](https://nextflow.io/). The easiest way to install Nextflow is by using conda.
   ```{bash}
   conda create -n nf-env -c bioconda nextflow
   ```

---
<br>

### 📄 Preparing Your Input Data 

Before you unleash EvOlf’s predictive power, make sure your input data is formatted correctly! Each row should represent a single ligand-receptor interaction and must include:  

1. **SMILES** ⌬ – The molecular representation of the ligand.    
    - **Important Note on SMILES Format** <br> 
 For optimal prediction accuracy, we strongly recommend converting your ligand SMILES to canonical SMILES format using OpenBabel before running EvOlf. This standardisation ensures consistency in molecular representation and significantly improves model performance.
      ```bash
      # Example of converting to canonical SMILES using OpenBabel
      obabel -ismi input.smi -ocan -O out.smi
      ```
3. **Receptor Sequence** 🪢 – The amino acid sequence of the GPCR.

To keep things organised, you can provide your own **unique identifiers** for ligands, receptors, and ligand-receptor pairs, or if you prefer, EvOlf can generate them for you automatically. 
Let’s go through both options! 🚀

---
<br> 

#### 📝 Simplest Format: No IDs Required!
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

#### 📝 Provide Unique Identifiers
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

#### ❌ Common Mistakes to Avoid <br>
Providing incorrect or inconsistent identifiers can cause errors in prediction. Below is an example of what NOT to do:

Bad Input Data Example
| ID | Ligand_ID | SMILES | Receptor_ID| Sequence |
|:-----|:---------|:-----------|:---------|:-----------|
|LR1  |L1  |OCC/C=C\CC  |R1  | MEPRKNVTDFVLLGFTQNPKEQKVLFVMFLLFYILTMVGNLLIVVTVTVSETLGSPMSFFLAGLTFIDIIYSSSISPRLISDLFFGNNSISFQSFMAQLFIEHLFGGSEVFLLLVMAYDRYVAICKPLHYLVIMRQWVCVLLLVVSWVGGFLQSVFQLSIIYGLPFCGPNVIDHFFCDMYPLLKLACTDTHVIGLLVVANGGLSCTIAFLLLLISYGVILHSLKKLSQKGRQKAHSTCSSHITVVVFFFVPCIFMCARPARTFSIDKSVSVFYTVITPMLNPLIYTLRNSEMTSAMKKL|
|LR2  |L2  |OCC/C=C\CC  |R1  | MIPIQLTVFFMIIYVLESLTIIVQSSLIVAVLGREWLQVRRLMPVDMILISLGISRFCLQWASMLNNFCSYFNLNYVLCNLTITWEFFNILTFWLNSLLTVFYCIKVSSFTHHIFLWLRWRILRLFPWILLGSLMITCVTIIPSAIGNYIQIQLLTMEHLPRNSTVTDKLENFHQYQFQAHTVALVIPFILFLASTIFLMASLTKQIQHHSTGHCNPSMKARFTALRSLAVLFIVFTSYFLTILITIIGTLFDKRCWLWVWEAFVYAFILMHSTSLMLSSPTLKRILKGKC|
|LR3  |L3  |CCCC(=O)O  |R2  | MGRGNSTEVTEFHLLGFGVQHEFQHVLFIVLLLIYVTSLIGNIGMILLIKTDSRLQTPMYFFPQHLAFVDICYTSAITPKMLQSFTEENNLITFRGCVIQFLVYATFATSDCYLLAIMAMDCYVAICKPLRYPMIMSQTVYIQLVAGSYIIGSINASVHTGFTFSLSFCKSNKINHFFCDGLPILALSCSNIDINIILDVVFVGFDLMFTELVIIFSYIYIMVTILKMSSTAGRKKSFSTCASHLTAVTIFYGTLSYMYLQPQSNNSQENMKVASIFYGTVIPMLNPLIYSLRNKEGK|
|LR4  |L3  |CCCCCCCC=O  |R3  | MGRGNSTEVTEFHLLGFGVQHEFQHVLFIVLLLIYVTSLIGNIGMILLIKTDSRLQTPMYFFPQHLAFVDICYTSAITPKMLQSFTEENNLITFRGCVIQFLVYATFATSDCYLLAIMAMDCYVAICKPLRYPMIMSQTVYIQLVAGSYIIGSINASVHTGFTFSLSFCKSNKINHFFCDGLPILALSCSNIDINIILDVVFVGFDLMFTELVIIFSYIYIMVTILKMSSTAGRKKSFSTCASHLTAVTIFYGTLSYMYLQPQSNNSQENMKVASIFYGTVIPMLNPLIYSLRNKEGK|
|LR4  |L4  |CC(=O)C(=O)C  |R4  | MVGANHSVVSEFVFLGLTNSWEIRLLLLVFSSMFYMASMMGNSLILLTVTSDPHLHSPMYFLLANLSFIDLGVSSVTSPKMIYDLFRKHEVISFGGCIAQIFFIHVIGGVEMVLLIAMAFDRYVAICKPLQYLTIMSPRMCMFFLVAAWVTGLIHSVVQLVFVVNLPFCGPNVSDSFYCDLPRFIKLACTDSYRLEFMVTANSGFISLGSFFILIISYVVIILTVLKHSSAGLSKALSTLSAHVSVVVLFFGPLIFVYTWPSPSTHLDKFLAIFDAVLTPVLNPIIYTFRN|

🚨 **What's Wrong Here?** <br>
- **Repeated Ligand IDs for Different SMILES** → "L3" is assigned to two different SMILES structures. Each ligand should have one unique ID. <br>
- **Different Ligand IDs for the Same SMILES** → "L1" and "L2" both have the same SMILES (OCC/C=C\CC). A single ligand should always have a consistent Ligand_ID. <br> 
- **Repeated Receptor IDs for Different Sequences** → "R1" is assigned to two different receptor sequences. Each receptor sequence should have one unique ID. <br> 
- **Different Receptor IDs for the Same Sequence** → "R2" and "R3" have the exact same receptor sequence, yet they are assigned different Receptor_IDs. Each unique receptor sequence should have one consistent Receptor_ID to avoid confusion. <br>
- **Duplicate Ligand-Receptor Pair IDs** → "LR4" appears twice. Every row should have a unique ID for proper tracking. <br> 

Data like this will break EvOlf’s ability to correctly associate ligands and receptors!

---
<br>

🛠 **Let EvOlf Handle the IDs!** <br>
If you don't want to manually assign identifiers. Just provide a CSV with ligand SMILES and receptor sequences, and EvOlf will take care of the rest. The assigned IDs will be available in the `Input_ID_Information.csv` file in the output directory.

---
<br>

#### 🎯 Key Takeaways
✔ Option 1: Provide only SMILES and Receptor Sequences, and let EvOlf handle the rest. <br> 
✔ Option 2: Provide custom unique IDs for ligands, receptors, and ligand-receptor pairs for better tracking. <br> 
✔ Make sure all IDs are unique to avoid errors in predictions. <br> 
✔ If you skip IDs, check `Input_ID_Information.csv` for automatically assigned identifiers. <br> 

Now you’re all set to get your predictions!🔎 <br> <br>

## 🏎️ Run Pipeline

1. Activate the conda environment containing NextFlow.
    ```{bash}
    conda activate nf-env
    ```
2. Run Nextflow command for evolf-pipeline, we provide two different ways to run the pipeline:
   1. Single File Mode: Use this for running a single dataset.
        ```
        nextflow run the-ahuja-lab/evolf-pipeline \
        --inputFile "data/my_experiment.csv" \
        --ligandSmiles "SMILES" \
        --receptorSequence "Sequence" \
        --outdir "./results/experiment_1" \
        -profile docker,gpu
        ```

   2. Batch Mode (High-Throughput): Use this to process multiple datasets in parallel. 
      - Create a manifest CSV file:
    
         batch_manifest.csv:
         ```
         inputFile,ligandSmiles,receptorSequence,ligandID,receptorID,lrID
         /data/project_A.csv,SMILES,Seq,L_ID,R_ID,Pair_ID
         /data/project_B.csv,cano_smiles,aa_seq,lig_name,prot_name,int_id
         ```
      - Run Command:
         ```
         nextflow run the-ahuja-lab/evolf-pipeline \
         --batchFile "batch_manifest.csv" \
         --outdir "./results/batch_run" \
         -profile docker
         ```

As the EvOlf pipeline uses NextFlow to run the EvOlf prediction model, it allows users to enable and disable some key features of the pipeline from the starting command itself. The options which are built into the EvOlf pipeline are as follows:

### Hardware Profiles (`-profile`)

  * **`docker`**: Uses Docker. Runs as the current user (`-u $(id -u)`) to prevent root-owned file issues.
  * **`apptainer` / `singularity`**: Uses `.sif` images. Automatically mounts your project directory.
  * **`gpu`**: **Highly Recommended.** Enables CUDA for ChemBERTa, ProtBERT, ProtT5, and the Prediction model. Without this, the pipeline will run on CPU (which is much slower).

### Model Caching

The first time you run EvOlf, it will download \~15GB of model weights (Hugging Face transformers).

  * **Location:** These are stored in `~/.evolf_cache` in your home directory.
  * **Benefit:** They are downloaded **only once**. All subsequent runs (even in different project folders) will use this centralised cache.

### Resource Management

The Nextflow pipeline is limited to utilising 4 CPU cores and 16GB RAM by default. You can change these defaults by providing the following parameters along with input:
- `--maxCPUs` to set the maximum number of CPU cores NextFlow can utilise to run EvOlf.
- `--maxMemory` to set the maximum amount of RAM the EvOlf pipeline can use.

> [!NOTE]
> For detailed documentation on all the options and profiles built into the EvOlf pipeline. Kindly visit its GitHub repo at [the-ahuja-lab/evolf-pipeline](https://www.github.com/the-ahuja-lab/evolf-pipeline)

## 📂 Output Files 
After running the pipeline, the EvOlf pipeline generates a separate folder for each input file in your output directory, with the following structure.
```
results/
├── my_experiment/
│   ├── my_experiment_Prediction_Output.csv    <-- FINAL RESULTS
│   ├── my_experiment_Input_ID_Information.csv <-- ID Mapping Key
│   └── pipeline_info/                           
│       ├── execution_report.html              <-- Resource usage (RAM/CPU)
│       └── execution_timeline.html            <-- Gantt chart of jobs
```

Each output file provides valuable insights into your data and predictions. Whether you're looking for interaction predictions or just need embeddings for your own model, EvOlf provides both. Here’s what each file means:

1️⃣ **Input_ID_Information.csv** <br>
📌 **What it contains**: <br_>
Unique IDs are assigned to each ligand, receptor, and ligand-receptor pair. <br> 
Information on any pairs that were not processed. <br> 
💡 **Why it matters**: This file helps track how your inputs were mapped and ensures all IDs are assigned correctly. <br>

✅ Example of `Input_ID_Information.csv`:
|SrNum|IDs|Ligand_ID|SMILES|Receptor_ID|Sequence|ProcessingStatus|
|:----|:-------|:---------|:----------|:---------|:----------|:-------|
|1|LR1|L1|OCC/C=C\CC|R1|MEPRKNVTDFVLLGFTQN...|Processed|
|2|LR2|L1|OCC/C=C\CC|R2|MIPIQLTVFFMIIYVLES...|Processed|
|3|LR3|L2|CCCC(=O)O|R3|MGRGNSTEVTEFHLLGFGV...|Processed|
|4|LR4|L3|CCCCCCCC=O|R3|MGRGNSTEVTEFHLLGFG...|Processed|
|5|LR5|L4|CC(=O)C(=O)C|R4|MVGANHSVVSEFVFLG...|Processed|

2️⃣**Ligand_Embeddings.csv** <br>
📌 **What it contains**: <br>
Numerical vector representations (embeddings) of all ligands generated by EvOlf. <br>
💡 **Why it matters**: Want to train your own model or analyse ligand properties? These embeddings provide a machine-readable format that captures key molecular features.

3️⃣**Receptor_Embeddings.csv** <br>
📌 **What it contains**: <br>
Embeddings for all receptors, similar to ligand embeddings but for receptor sequences. <br>
💡 **Why it matters**: Useful for training models or clustering receptors.

4️⃣ LR_Pair_Embeddings.csv <br>
📌 **What it contains**: <br>
Combined ligand-receptor pair embeddings. <br>
💡 **Why it matters**: These embeddings power EvOlf’s interaction predictions, but you can also use them for custom machine-learning applications!

5️⃣ Prediction_Output.csv <br>
📌 **What it contains**: <br>
The final predictions made by EvOlf include interaction labels (0 or 1) and confidence scores. <br>
💡 **Why it matters**: This file tells you which ligand-receptor pairs are predicted to interact and with what level of confidence.

✅ Example of `Prediction_Output.csv`:
|ID|Predicted Label|P1|
|:-------|:---------|:----------|
|LR1|0|0.000226742238737642|
|LR2|0|0.000139558469527401|
|LR3|0|0.00170087465085089|
|LR4|0|0.000176260960870422|
|LR5|0|0.12915557622909546|
