import pandas as pd
from signaturizer import Signaturizer
from rdkit import Chem, RDLogger
import numpy as np
import os
import pickle
import argparse
import pyreadr

RDLogger.DisableLog('rdApp.*')

sign = None
colNames = None

def get_model():
    global sign, colNames
    
    # This 'if' block will only run ONCE
    if sign is None:
        print("--- Loading Signaturizer model (lazy)... ---")
        sign = Signaturizer('GLOBAL', verbose=False)
        
        # Generate your colNames here
        colNames = []
        for alphabet in "ABCDE":
          for i in range(1, 6):
            for j in range(1, 129):
              colNames.append(f"{alphabet}{i}_{j}")
        print("--- Model loaded. ---")
        
    return sign, colNames

print(colNames)
    
# Codes Modified by me:
def SMILESCheck(dat):
    # initial number of smiles present in the datafile
    print('Initial count - '+str(len(dat)))
    final_smiles = []
    data = pd.DataFrame()
    print('Checking SMILES')
    for i in dat['SMILES']:
        ms = Chem.MolFromSmiles(i) # will create a molecular object for that smile
        if ms is not None: # if the smiles is successfully converted
            try:
                inchi = Chem.rdinchi.MolToInchi(ms)[0] # will find the InChi key of that smiles from the molecular object
                final_smiles.append(i)
            except: # if inchi is not calculated
                print('Skipping, Kekulize error.')
        if ms is None: # if the smiles is not successfully converted
            print('Skipping, NoneMol.')
    data['SMILES'] = final_smiles # final data frame with the list of smiles that are valid in nature
    print('After CleanUp - ' + str(len(data)))
    return data


def FeatureSignaturizer(dat):
    print('Calculating Signaturizer Emdeddings')
    
    sign_model, model_col_names = get_model()
    
    # add the smiles to the final df
    sig_df = pd.DataFrame()
    sig_df['SMILES'] = dat['SMILES'].tolist()

    # calculate embeddings using Signaturizer
    results = sign_model.predict(dat['SMILES'].tolist())
    print('Emdeddings Calculated')

    # convert the embeddings as a dataframe
    df = pd.DataFrame(results.signature)
    # rename the columns to desired names
    df.columns = model_col_names
    # combine smiles and embeddings to a single data frame
    sig_df = pd.concat([sig_df,df],axis = 1)
    
    del results
    del df
    
    return sig_df



def HandleMissingValues(DataFile):
    print('Processing Missing Values.')
    data = DataFile.drop(['SMILES'],axis=1)
    
    # replace spaces and infinities with NaN
    data = data.replace([np.inf, -np.inf, "", " "], np.nan)
    print('Replaced spaces and inf with NaN')
    
    # find index of rows with all NaN values
    mask = data.isna().all(axis=1)
    idx = data.index[mask]

    # drop rows with all NaN values
    data = data.drop(index = idx)
    DataFile = DataFile.drop(index = idx)
    
    print('Dropped rows with all NaNs')
    
    # replace the nans with column mean
    for i in data.columns:
        data[i] = data[i].fillna(data[i].mean())
    print('Imputed Missing Values with Mean')
    
    # add the SMILES back in as the first column
    data = pd.concat([DataFile['SMILES'], data], axis=1)
    
    del DataFile
    
    print('Final count - '+str(len(data)))
    return data


def signaturizer(input_file: str, output_file: str):
    RawData = pd.read_csv(input_file)
    RawData


    print(len(RawData.SMILES.unique()))

    # check for valid smiles
    CleanData = SMILESCheck(RawData)

    # Compute Signaturizer embeddings
    Signatures = FeatureSignaturizer(CleanData)

    # Handle missing values
    Signatures = HandleMissingValues(Signatures)


    # Add Ligand IDs 
    # Signatures = pd.concat([RawData['Ligand_ID'], Signatures], axis=1)
    # # check if SMILES are same in both the files
    # if Signatures['SMILES'].equals(RawData['SMILES']):
    #     Signatures = pd.concat([RawData['Ligand_ID'], Signatures], axis=1)

    # Merge Signatures with Ligand_IDs based on SMILES
    Signatures = Signatures.merge(RawData[['SMILES', 'Ligand_ID']], on='SMILES', how='left')
    Signatures

    # Move Ligand_ID to the first column for readability
    cols = ['Ligand_ID'] + [col for col in Signatures.columns if col != 'Ligand_ID']
    Signatures = Signatures[cols]
    Signatures

    # Save the embeddings as csv
    Signatures.to_csv(output_file, index = False)

    print("Code ran successfully")


parser = argparse.ArgumentParser(description="Signaturizer")
parser.add_argument("--input", help="Input File Path")
parser.add_argument("--output", help="Output File Path", default="Raw_Signaturizer.csv")

args = parser.parse_args()

signaturizer(args.input, args.output)