// Define the FEATURE_COMPILATION process to compile features
// from various featurizers into a final dataset for prediction.
// It takes the individual feature files along with the main data
// and receptor CSV from PREPARE_INPUT, and outputs a compiled feature file.
process FEATURE_COMPILATION {    
	label 'EVOLFR'

    publishDir "${params.outdir}/${meta_id}", mode: 'copy', pattern: "${meta_id}_Input_ID_Information.csv"
    
    // Define input parameters
    input:
    tuple val(meta_id), path(receptor_csv_ch), path(main_data_ch), path(signaturizer_ch), path(chemberta_ch), path(mol2vec_ch), path(graph2vec_ch), path(mordred_ch), path(protr_ch), path(prott5_ch), path(protbert_ch), path(mf02_ch), path(mf04_ch), path(mf06_ch), path(mf08_ch), path(mf09_ch), path(mf10_ch), path(mf11_ch)
    
    // Define output channels
    // Compile all features into a single output channel
    output:
    path "${meta_id}_Input_ID_Information.csv"

    tuple val(meta_id), \
    path("${meta_id}_mainData.csv"), \
    path("${meta_id}_Graph2Vec_Final.csv"), \
    path("${meta_id}_Signaturizer_Final.csv"), \
    path("${meta_id}_Mordred_Final.csv"), \
    path("${meta_id}_Mol2Vec_Final.csv"), \
    path("${meta_id}_ChemBERTa_Final.csv"), \
    path("${meta_id}_ProtR_Final.csv"), \
    path("${meta_id}_ProtT5_Final.csv"), \
    path("${meta_id}_ProtBERT_Final.csv"), \
    path("${meta_id}_MathFeature_Final.csv"), \
    emit: fc_all_ch

    script:
    """
    conda run -n evolfr_env Rscript /app/FeatureCompilation.R \
        ${receptor_csv_ch} \
        ${main_data_ch} \
        ${signaturizer_ch} \
        ${chemberta_ch} \
        ${mol2vec_ch} \
        ${graph2vec_ch} \
        ${mordred_ch} \
        ${protr_ch} \
        ${prott5_ch} \
        ${protbert_ch} \
        ${mf02_ch} \
        ${mf04_ch} \
        ${mf06_ch} \
        ${mf08_ch} \
        ${mf09_ch} \
        ${mf10_ch} \
        ${mf11_ch} \
        ${meta_id}
    """
}
