// Define the SIGNATURIZER process to compute Signaturizer features
// for ligand molecules based on their SMILES representations.
// It takes the standardized ligand CSV file from PREPARE_INPUT
// and outputs a channel with Signaturizer feature files.
process SIGNATURIZER {
    // Set input parameters
    input:
    tuple val(meta_id), path(ligands_csv)

    // Define output channels
    output:
    tuple val(meta_id), path("${meta_id}_Raw_Signaturizer.csv"), emit: signaturizer_ch

    script:
    """
    conda run -n signaturizer_env python /app/Signaturizer.py --input ${ligands_csv} --output "${meta_id}_Raw_Signaturizer.csv"
    """
}