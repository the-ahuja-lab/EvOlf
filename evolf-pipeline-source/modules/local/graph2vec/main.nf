// Define the GRAPH2VEC process to generate graph embeddings
// for ligand molecules based on their molecular graphs.
// It takes the standardized ligand CSV file from PREPARE_INPUT
// and outputs a channel with Graph2Vec feature files.

process GRAPH2VEC {
    // Set input parameters
    input:
    tuple val(meta_id), path(input_file)
    
    // Define output channels
    output:
    tuple val(meta_id), path("${meta_id}_Raw_Graph2Vec.csv"), emit: graph2vec_ch

    script:
    """
    conda run -n graph2vec_env python /app/Graph2Vec.py --input ${input_file} --output "${meta_id}_Raw_Graph2Vec.csv"
    """
}
