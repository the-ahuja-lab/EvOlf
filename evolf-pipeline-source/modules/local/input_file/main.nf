// Define the PREPARE_INPUT process to standardize input files
// into a common format for downstream processing.
// This process will handle both single file and batch file modes.
// It outputs standardized channels for ligands and receptors.
process PREPARE_INPUT {
    label 'EVOLFR'
    // Define input parameters
    input:
    tuple val(meta_id), path(input_file), val(smiles_col), val(seq_col), val(lig_id_col), val(rec_id_col), val(pair_id_col)
    
    // Define output channels
    output:
    tuple val(meta_id), path("mainData_01.csv"), emit: main_data_ch
    tuple val(meta_id), path("ligsData.csv"), emit: ligands_csv_ch
    tuple val(meta_id), path("recsData.csv"), emit: receptors_csv_ch
    tuple val(meta_id), path("recsData.fasta"), emit: receptors_fasta_ch

    script:
    """
    conda run -n evolfr_env \
    Rscript /app/InputFile.R \
            ${input_file} \
            ${smiles_col} \
            ${seq_col} \
            ${lig_id_col} \
            ${rec_id_col} \
            ${pair_id_col}
    """
}
