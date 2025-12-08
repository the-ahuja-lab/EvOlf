// Define the PROTT5 process to generate ProtT5 embeddings
// for receptor sequences based on their FASTA representations.
// It takes the standardized receptor FASTA file from PREPARE_INPUT
// and outputs a channel with ProtT5 feature files.
process PROTT5 {
    label 'EVOLFDL'
    
    // Define input parameters
    input:
    tuple val(meta_id), path(input_file)
    path hf_cache

    // Define output channels
    output:
    tuple val(meta_id), path("${meta_id}_Raw_ProtT5.csv"), emit: prott5_ch

    script:
    """
    set +u
    source /opt/conda/etc/profile.d/conda.sh
    conda activate evolfdl_env
    set -u
    export HF_HOME="${hf_cache}"
    export TRANSFORMERS_CACHE="${hf_cache}"
    python /app/ProtT5.py --input ${input_file} --output "${meta_id}_Raw_ProtT5.csv"
    """
}
