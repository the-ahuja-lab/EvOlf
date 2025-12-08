// Define the CHEMBERTA process to generate ChemBERTa features
// for ligand molecules based on their SMILES representations.
// It takes the standardized ligand CSV file from PREPARE_INPUT
// and outputs a channel with ChemBERTa feature files.

process CHEMBERTA {
    label 'EVOLFDL'    
    // Define input parameters
    input:
    tuple val(meta_id), path(input_file)
    path hf_cache

    // Define output channels
    output:
    tuple val(meta_id), path("${meta_id}_Raw_ChemBERTa.csv"), emit: chemberta_ch

    // Define the script to run ChemBERTa feature extraction
    script:
    """
    set +u
    source /opt/conda/etc/profile.d/conda.sh
    conda activate evolfdl_env
    set -u
    export HF_HOME="${hf_cache}"
    export TRANSFORMERS_CACHE="${hf_cache}"
    python /app/ChemBERTa.py --input ${input_file} --output "${meta_id}_Raw_ChemBERTa.csv"
    """
}
