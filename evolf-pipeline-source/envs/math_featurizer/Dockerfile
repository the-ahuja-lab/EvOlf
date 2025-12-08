FROM condaforge/miniforge3

WORKDIR /app

COPY envs/math_featurizer/MathFeaturizer.yml .

RUN conda env create -f MathFeaturizer.yml -n mathfeaturizer_env && conda clean -afy

COPY envs/math_featurizer/MathFeaturizer.py .

COPY envs/math_featurizer/methods/ /app/methods/
