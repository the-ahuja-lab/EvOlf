// Define the EVOLF_PREDICTION process to perform
// predictions based on compiled features from various featurizers.
// It takes the final main data and all feature channels
// produced by FEATURE_COMPILATION and outputs prediction results
// along with ligand, receptor, and LR pair embeddings.
process EVOLF_PREDICTION {
    // Set the publish directory for output files
    publishDir "${params.outdir}/${meta_id}", mode: 'copy'
    
    // Define input parameters
    input:
    tuple val(meta_id), path(final_main_data), path(fc_graph2vec), path(fc_signaturizer), path(fc_mordred), path(fc_mol2vec), path(fc_chemberta), path(fc_protr), path(fc_prott5), path(fc_protbert), path(fc_mathfeature)
    
    // Define output channels
    output:
    path "${meta_id}_Prediction_Output.csv", emit: prediction_output_ch
    path "${meta_id}_Ligand_Embeddings.csv", emit: ligand_embeddings_ch
    path "${meta_id}_Receptor_Embeddings.csv", emit: receptor_embeddings_ch
    path "${meta_id}_LR_Pair_Embeddings.csv", emit: lr_pair_embeddings_ch

    // Define the script to run the EvOlf prediction
    // using the provided feature files
    // and output the results to designated files.
    // The script sets up necessary environment variables
    // for matplotlib caching to avoid runtime issues.
    script:
    """
    export MPLCONFIGDIR="./.matplotlib_cache"
    export XDG_CACHE_HOME="./.xdg_cache"

    conda run -n prediction_env python /app/EvOlfPrediction.py \
        --final_main_data ${final_main_data} \
        --fc_graph2vec ${fc_graph2vec} \
        --fc_signaturizer ${fc_signaturizer} \
        --fc_mordred ${fc_mordred} \
        --fc_mol2vec ${fc_mol2vec} \
        --fc_chemberta ${fc_chemberta} \
        --fc_protbert ${fc_protbert} \
        --fc_prott5 ${fc_prott5} \
        --fc_mathfeature ${fc_mathfeature} \
        --fc_protr ${fc_protr} \
        --output_prefix ${meta_id}
    """   
}