// Define the MOL2VEC process to generate molecular embeddings
// for ligand molecules based on their SMILES representations.
// It takes the standardized ligand CSV file from PREPARE_INPUT
// and outputs a channel with Mol2Vec feature files.

process MOL2VEC {
    // Set input parameters
    input:
    tuple val(meta_id), path(input_file)

    // Define output channels
    output:
    tuple val(meta_id), path("${meta_id}_Raw_Mol2Vec.csv"), emit: mol2vec_ch

    script:
    """
    conda run -n mol2vec_env python /app/Mol2Vec.py --input ${input_file} --output "${meta_id}_Raw_Mol2Vec.csv"
    """
} 
