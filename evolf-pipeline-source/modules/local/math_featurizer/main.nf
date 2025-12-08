// Define the MATH_FEATURIZER process to compute mathematical features
// for receptor sequences based on their FASTA representations.
// It takes the standardized receptor FASTA file from PREPARE_INPUT
// and outputs a channel with multiple MathFeature files.
process MATH_FEATURIZER {
    // Set input parameters
    input:
    tuple val(meta_id), path(input_file)

    // Define output channels
    output:
    tuple val(meta_id), \
          path("${meta_id}_MF_02.csv"), \
          path("${meta_id}_MF_04.csv"), \
          path("${meta_id}_MF_06.csv"), \
          path("${meta_id}_MF_08.csv"), \
          path("${meta_id}_MF_09.csv"), \
          path("${meta_id}_MF_10.csv"), \
          path("${meta_id}_MF_11.csv"), \
          emit: mf_all_ch

    script:
    """
    conda run -n mathfeaturizer_env python /app/MathFeaturizer.py --input ${input_file} --output_prefix ${meta_id}
    """
}
