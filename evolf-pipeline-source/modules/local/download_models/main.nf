process DOWNLOAD_MODELS {
    label 'EVOLFDL'

    storeDir "${params.model_cache_dir}"

    output:
    path "hf_cache", emit: cache_ch

    script:
    """
    mkdir -p hf_cache
    
    export HF_HOME=\$(readlink -f hf_cache)
    export TRANSFORMERS_CACHE=\$(readlink -f hf_cache)
    set +u
    source /opt/conda/etc/profile.d/conda.sh
    conda activate evolfdl_env
    set -u
    python -c "
    import os
    from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer, BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer; \
    import torch; \

    # Set the cache directory for Hugging Face models
    cache_dir = os.environ['HF_HOME']

    print(f'Downloading models to {cache_dir}...')
    
    model_version = 'DeepChem/ChemBERTa-77M-MLM';\
    model = RobertaModel.from_pretrained(model_version, output_attentions=True, cache_dir=cache_dir);\
    tokenizer = RobertaTokenizer.from_pretrained(model_version, cache_dir=cache_dir);\

    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, cache_dir=cache_dir);\
    model = BertModel.from_pretrained('Rostlab/prot_bert_bfd', cache_dir=cache_dir);\
    model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', cache_dir=cache_dir);\
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False, cache_dir=cache_dir);

    print('All downloads complete.')
    "
    """
}