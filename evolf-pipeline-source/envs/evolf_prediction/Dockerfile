# 1. Start from a base image
FROM continuumio/miniconda3
WORKDIR /app

# 2. Copy the environment file
COPY envs/evolf_prediction/Prediction.yml .

# 3. Create the Conda environment
RUN conda env create -f Prediction.yml -n prediction_env && conda clean -afy

# Copy all the code
COPY envs/evolf_prediction/EvOlfPrediction.py .
COPY envs/evolf_prediction/architecture.py .
COPY envs/evolf_prediction/myDataset.py .
COPY envs/evolf_prediction/myImports.py .
COPY envs/evolf_prediction/myTrainParams.py .

# Copy all the models
COPY assets/models/ /app/models/
