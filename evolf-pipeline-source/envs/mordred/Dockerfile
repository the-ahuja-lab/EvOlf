# 1. Start from a base image
FROM continuumio/miniconda3
WORKDIR /app

# 2. Copy the environment file
COPY envs/mordred/Mordred.yml .

# 3. Create the Conda environment
RUN conda env create -f Mordred.yml -n mordred_env && conda clean -afy

# 4. Copy your application code
COPY envs/mordred/Mordred.py .  

# 5. Copy the model
COPY assets/22_MordredFinalColumns.csv /app/models/
