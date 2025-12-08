# 1. Start from a base image
FROM condaforge/miniforge3
WORKDIR /app

# 2. Copy the environment file
COPY envs/graph2vec/Graph2Vec.yml .

# 3. Create the Conda environment
RUN conda env create -f Graph2Vec.yml -n graph2vec_env && conda clean -afy

# 4. Copy your application code
COPY envs/graph2vec/Graph2Vec.py .

# 5. Copy the model
COPY assets/Common_Input_G2V_15.csv /app/models/
