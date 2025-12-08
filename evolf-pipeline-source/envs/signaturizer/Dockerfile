# Start from Conda
FROM condaforge/miniforge3
WORKDIR /app

# 1. Copy the YML file from its envs folder
COPY envs/signaturizer/Signaturizer.yml .

# 2. Create the environment
RUN conda env create -f Signaturizer.yml -n signaturizer_env && conda clean -afy

# 3. Burn the signaturizer model in image
RUN conda run -n signaturizer_env python -c "\
print('--- Pre-warming Signaturizer cache... ---'); \
import rdkit; \
from signaturizer import Signaturizer; \
sign = Signaturizer('GLOBAL', verbose=False); \
print('--- Signaturizer cache built. ---')"

# 3. Copy the Python script
COPY envs/signaturizer/Signaturizer.py .
