// Define the PROTR process to compute ProtR features
// for receptor sequences based on their FASTA representations.
// It takes the standardized receptor FASTA file from PREPARE_INPUT
// and outputs a channel with ProtR feature files.
process PROTR {
    label 'EVOLFR'
    
    // Define input parameters
    input:
    tuple val(meta_id), path(input_file)

    // Define output channels
    output:
    tuple val(meta_id), path("${meta_id}_Raw_ProtR.csv"), emit: protr_ch

    script:
    """
    conda run -n evolfr_env \
    Rscript /app/ProtR.R \
            ${input_file} \
            "${meta_id}_Raw_ProtR.csv"
    """
}
