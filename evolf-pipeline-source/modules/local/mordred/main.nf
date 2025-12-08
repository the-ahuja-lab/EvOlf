// Define the MORDRED process to compute molecular descriptors
// for ligand molecules based on their SMILES representations.
// It takes the standardized ligand CSV file from PREPARE_INPUT
// and outputs a channel with Mordred feature files.
process MORDRED {
    // Set input parameters
    input:
    tuple val(meta_id), path(input_file)

    // Define output channels
    output:
    tuple val(meta_id), path("${meta_id}_Raw_Mordred.csv"), emit: mordred_ch

    script:
    """
    conda run -n mordred_env python /app/Mordred.py --input ${input_file} --output "${meta_id}_Raw_Mordred.csv"
    """
}
