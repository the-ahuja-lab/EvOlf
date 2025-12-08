# Start from Conda
FROM condaforge/miniforge3
WORKDIR /app

# 1. Copy the YML file from its envs folder
COPY envs/EvOlf_DL/EvOlf_DL.yml .

# 2. Create the environment
RUN conda env create -f EvOlf_DL.yml -n evolfdl_env && \
    conda clean -afy && \
    # This line cleans the pip cache inside the new env
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.pyc' -delete && \
    rm -rf /root/.cache/pip

# 3. Copy the Python script
COPY envs/EvOlf_DL/*.py .
