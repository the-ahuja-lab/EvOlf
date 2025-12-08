# Start from Conda
FROM condaforge/miniforge3
WORKDIR /app

# 1. Copy the YML file from its envs folder
COPY envs/EvOlf_R/EvOlf_R.yml .

# 2. Create the environment
RUN conda env create -f EvOlf_R.yml -n evolfr_env && conda clean -afy


# 3. Copy the Python script
COPY envs/EvOlf_R/*.R .
