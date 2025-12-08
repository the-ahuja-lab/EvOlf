/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT FUNCTIONS / MODULES / SUBWORKFLOWS / WORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
include { DOWNLOAD_MODELS } from "./modules/local/download_models/main.nf"
include { PREPARE_INPUT } from "./modules/local/input_file/main.nf"
include { SIGNATURIZER } from "./modules/local/signaturizer/main.nf"
include { CHEMBERTA } from "./modules/local/chemberta/main.nf"
include { MOL2VEC } from "./modules/local/mol2vec/main.nf"
include { MORDRED } from "./modules/local/mordred/main.nf"
include { GRAPH2VEC } from "./modules/local/graph2vec/main.nf"
include { PROTR } from "./modules/local/protr/main.nf"
include { PROTT5 } from "./modules/local/prott5/main.nf"
include { PROTBERT } from "./modules/local/protbert/main.nf"
include { MATH_FEATURIZER } from "./modules/local/math_featurizer/main.nf"
include { FEATURE_COMPILATION } from "./modules/local/feature_compilation/main.nf"
include { EVOLF_PREDICTION } from "./modules/local/evolf_prediction/main.nf"

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOW FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

// WORKFLOW: Run main nf-core/sarek analysis pipeline

workflow {

    // --- INPUT CHANNEL SETUP ---
    // Create a standardized input channel based on whether running in BATCH or SINGLE file mode.
    input_params_ch = params.batchFile ?
    (
        {
            // --- BATCH MODE (if params.batchFile is provided) ---
            log.info "Running in BATCH mode. Using samplefile: ${params.batchFile}"
            // Read the batch file (CSV) and create a channel of tuples
            return Channel.fromPath(params.batchFile).splitCsv(header: true)
                .map { row ->
                    // Resolve the file path from the CSV
                    def input_file = file(row.inputFile)
                    // Get the meta_id from the file name 
                    def meta_id = input_file.simpleName
                    if (!input_file.exists()) {
                        log.warn "Input file not found, skipping: ${row.inputFile}"
                        return null // Skip this row
                    }
                    // Create a tuple matching the expected input for PREPARE_INPUT
                    return [ meta_id, input_file, row.ligandSmiles, row.receptorSequence, row.ligandID, row.receptorID, row.lrID ]
                }
                .filter { it != null } // Remove any skipped (null) entries
        }() // <-- The () here immediately executes the closure
    ) :
    // --- SINGLE FILE MODE (if params.batchFile is false/null) ---
    (
        {
            // Validate that inputFile is provided
            if (!params.inputFile) {
                exit 1, "Please provide --inputFile (for single run) or --batchFile (for batch run)"
            }
            // Log the mode
            log.info "Running in SINGLE file mode. Using file: ${params.inputFile}"
            
            // Resolve the input file
            def input_data_file = file(params.inputFile)
            if (!input_data_file.exists()) {
                    exit 1, "Input file not found: ${params.inputFile}"
            }
            // Get the meta_id from the file name
            def meta_id = input_data_file.simpleName
            // Create a single-item channel
            return channel.of([ meta_id, input_data_file, params.ligandSmiles, params.receptorSequence, params.ligandID, params.receptorID, params.lrID ])
        }() // <-- The () here immediately executes the closure
    )

    // --- PIPELINE EXECUTION ---

    // Download models if not already cached
    // This process will run only once at the start of the pipeline.
    DOWNLOAD_MODELS()

    // PREPARE_INPUT now takes the standardized channel, whether it has 1 or N items.
    PREPARE_INPUT(input_params_ch)

    // FEATURIZERS
    // Each featurizer module takes the relevant output channel from PREPARE_INPUT
    // and produces its own feature channel.

    SIGNATURIZER(PREPARE_INPUT.out.ligands_csv_ch)

    CHEMBERTA(PREPARE_INPUT.out.ligands_csv_ch, DOWNLOAD_MODELS.out.cache_ch)

    MOL2VEC(PREPARE_INPUT.out.ligands_csv_ch)

    MORDRED(PREPARE_INPUT.out.ligands_csv_ch)

    GRAPH2VEC(PREPARE_INPUT.out.ligands_csv_ch)

    PROTR(PREPARE_INPUT.out.receptors_csv_ch)

    PROTT5(PREPARE_INPUT.out.receptors_fasta_ch, DOWNLOAD_MODELS.out.cache_ch)

    PROTBERT(PREPARE_INPUT.out.receptors_csv_ch, DOWNLOAD_MODELS.out.cache_ch)

	MATH_FEATURIZER(PREPARE_INPUT.out.receptors_fasta_ch)
    
    // These feature channels are then combined for final compilation.
    // Create a combined features channel by joining all individual feature channels
    combined_features_ch = PREPARE_INPUT.out.receptors_csv_ch
                                .join(PREPARE_INPUT.out.main_data_ch)
                                .join(SIGNATURIZER.out.signaturizer_ch)
                                .join(CHEMBERTA.out.chemberta_ch)
                                .join(MOL2VEC.out.mol2vec_ch)
                                .join(GRAPH2VEC.out.graph2vec_ch)
                                .join(MORDRED.out.mordred_ch)
                                .join(PROTR.out.protr_ch)
                                .join(PROTT5.out.prott5_ch)
                                .join(PROTBERT.out.protbert_ch)
                                .join(MATH_FEATURIZER.out.mf_all_ch)
    
    // Pass the combined features channel to FEATURE_COMPILATION
    FEATURE_COMPILATION(combined_features_ch)
    
    // EVOLF PREDICTION
    // Finally, run the prediction module using the compiled features
    // from FEATURE_COMPILATION.
    EVOLF_PREDICTION(FEATURE_COMPILATION.out.fc_all_ch)
}
