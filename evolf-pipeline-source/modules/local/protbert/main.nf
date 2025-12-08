// Define the PROTBERT process to generate ProtBERT embeddings
// for receptor sequences based on their FASTA representations.
// It takes the standardized receptor FASTA file from PREPARE_INPUT
// and outputs a channel with ProtBERT feature files.

process PROTBERT {
    label 'EVOLFDL'
    
    // Define input parameters    
    input:
    tuple val(meta_id), path(input_file)
    path hf_cache

    // Define output channels
    output:
    tuple val(meta_id), path("${meta_id}_Raw_ProtBERT.csv"), emit: protbert_ch

    script:
    """
    set +u
    source /opt/conda/etc/profile.d/conda.sh
    conda activate evolfdl_env
    set -u
    export HF_HOME="${hf_cache}"
    export TRANSFORMERS_CACHE="${hf_cache}"
    python /app/ProtBERT.py --input ${input_file} --output "${meta_id}_Raw_ProtBERT.csv"
    """
}
