# import libraries
from mordred import Calculator, descriptors
import pandas as pd
import pickle
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
import os
import argparse

calc = None
final_cols = None

def get_resources():
    global calc, final_cols
    if calc is None:
        calc = Calculator(descriptors, ignore_3D=False)
        # load the list of final columns
        final_cols = pd.read_csv('/app/models/22_MordredFinalColumns.csv')
    return calc, final_cols

def Mordred(input_file: str, output_file: str):
    
    calc, finalCols = get_resources()
    # Load the Data
    RawData = pd.read_csv(input_file)
    
    smiles_list = RawData["SMILES"]

    # Convert the smiles into sdf to get the 3d coordinates as well

    invalid_smiles_list = []  # List to store invalid SMILES

    # Create an SDF writer
    w = Chem.SDWriter('Mordred_valid_SMILES.sdf')

    # Iterate over the SMILES list
    for smiles in smiles_list:

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
                invalid_smiles_list.append(smiles)  # Add the invalid SMILES to the list
        else:
            invalid_smiles_list.append(smiles)  # Add the invalid SMILES to the list

    # Close the SDF writer
    w.close()

    # save the smiles that failed
    # save the list of final columns
    with open('Mordred_invalid_SMILES.pickle', 'wb') as file:
        pickle.dump(invalid_smiles_list, file)


    # Drop rows with invalid SMILES from the "RawData" DataFrame
    FilteredData = RawData[~RawData['SMILES'].isin(invalid_smiles_list)]

    # read the sdf file:
    sdf_list = Chem.SDMolSupplier("Mordred_valid_SMILES.sdf")

    # Calculate the Mordred Descriptors
    featuresAll = calc.pandas(sdf_list)

    # convert all columns into numeric. This will introduce NaNs where the values are not numeric
    featuresAll = featuresAll.apply(pd.to_numeric, errors='coerce')
    # find the percentage of missing value in each column
    missingCount = featuresAll.isnull().mean() * 100


    # convert to a list
    finalCols = finalCols['0'].values.tolist()


    # subset the final columns from the descriptors
    featuresFiltered = featuresAll.filter(finalCols)

    # replace the missing values with column mean
    featuresFiltered = featuresFiltered.fillna(featuresFiltered.mean())

    # Add Ligand IDs and SMILES after checking if the number of datapoints are same in both the files
    featuresFiltered = pd.concat([FilteredData[['Ligand_ID', 'SMILES']], featuresFiltered], axis=1)

    featuresFiltered.to_csv(output_file, index = False)

    print("Code ran successfully")


parser = argparse.ArgumentParser(description="Mordred")
parser.add_argument("--input", help="Input File Path")
parser.add_argument("--output", help="Output File Path", default="Raw_Mordred.csv")

args = parser.parse_args()

Mordred(args.input, args.output)