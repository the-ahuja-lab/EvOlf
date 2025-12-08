# 1. Start from a base image
FROM continuumio/miniconda3
WORKDIR /app

# 2. Copy the environment file
COPY envs/mol2vec/Mol2Vec.yml .

# 3. Create the Conda environment
RUN conda env create -f Mol2Vec.yml -n mol2vec_env

RUN conda run -n mol2vec_env pip install --no-deps git+https://github.com/samoturk/mol2vec && conda clean -afy

# 4. Copy your application code
COPY envs/mol2vec/Mol2Vec.py .

# 5. Copy the model
COPY assets/model_300dim.pkl /app/models/

