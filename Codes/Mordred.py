# install packages
#! pip install mordred rdkit

# import libraries
from mordred import Calculator, descriptors
import pandas as pd
import pickle
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
import os

calc = Calculator(descriptors, ignore_3D=False)

# define the input and output paths
outPath = '/content/drive/MyDrive/PhD/Projects/eNOSE/EvOlf_3.3/15_Colab/'
dataPath = '/content/drive/MyDrive/PhD/Projects/eNOSE/EvOlf_3.3/15_Colab/'
csv_path=''

# Load the Data
RawData = pd.read_csv(dataPath+"ligsData.csv")
RawData

smiles_list = RawData["SMILES"]

# Convert the smiles into sdf to get the 3d coordinates as well

invalid_smiles_list = []  # List to store invalid SMILES

# Create an SDF writer
w = Chem.SDWriter(outPath+'Mordred_valid_SMILES.sdf')

# Iterate over the SMILES list
for smiles in smiles_list:
    # print(smiles)

    # Convert SMILES to RDKit Mol object
    mol = Chem.MolFromSmiles(smiles)

    # Check if conversion is successful
    if mol is not None:
        # Add explicit hydrogen atoms to the molecule
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol)

        # Check if embedding is successful
        if mol.GetNumConformers() > 0:
            # Write the Mol object to the SDF file
            w.write(mol)
        else:
            print("Embedding failed for SMILES:", smiles)
            invalid_smiles_list.append(smiles)  # Add the invalid SMILES to the list
    else:
        print("Conversion failed for SMILES:", smiles)
        invalid_smiles_list.append(smiles)  # Add the invalid SMILES to the list

# Close the SDF writer
w.close()

# save the smiles that failed
# save the list of final columns
with open(outPath+'Mordred_invalid_SMILES.pickle', 'wb') as file:
    pickle.dump(invalid_smiles_list, file)


# Drop rows with invalid SMILES from the "RawData" DataFrame
FilteredData = RawData[~RawData['SMILES'].isin(invalid_smiles_list)]
print(FilteredData.shape)
FilteredData

# read the sdf file:
sdf_list = Chem.SDMolSupplier(outPath+"Mordred_valid_SMILES.sdf")

# Calculate the Mordred Descriptors
featuresAll = calc.pandas(sdf_list)

# convert all columns into numeric. This will introduce NaNs where the values are not numeric
featuresAll = featuresAll.apply(pd.to_numeric, errors='coerce')
# find the percentage of missing value in each column
missingCount = featuresAll.isnull().mean() * 100
# find out how many columns have missing values more than 50%
print(len(missingCount[missingCount>50]))


# load the list of final columns
finalCols = pd.read_csv(csv_path+'22_MordredFinalColumns.csv')
# convert to a list
finalCols = finalCols['0'].values.tolist()

print(len(finalCols))
finalCols[0:6]


# subset the final columns from the descriptors
featuresFiltered = featuresAll.filter(finalCols)

# replace the missing values with column mean
featuresFiltered = featuresFiltered.fillna(featuresFiltered.mean())

# Add Ligand IDs and SMILES after checking if the number of datapoints are same in both the files
featuresFiltered = pd.concat([FilteredData[['Ligand_ID', 'SMILES']], featuresFiltered], axis=1)
featuresFiltered.head()

featuresFiltered.to_csv(outPath+"Raw_Mordred.csv", index = False)

print("Code ran successfully")