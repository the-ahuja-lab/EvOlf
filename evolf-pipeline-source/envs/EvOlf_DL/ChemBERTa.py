from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer
import torch
import pandas as pd
import argparse
import numpy as np
import random
import os

# Set a seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# These two lines are the most important
# This tells cuDNN to use *only* deterministic algorithms
torch.backends.cudnn.deterministic = True
# This disables the "benchmark" mode
torch.backends.cudnn.benchmark = False

model = None
tokenizer = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Using device: {device} ---")

def get_model():
    # download the pretrained model
    model_version = 'DeepChem/ChemBERTa-77M-MLM'
    
    global model, tokenizer
    
    # download and load the tokenizer which is used for pretraining the above model
    if model is None:
        model = RobertaModel.from_pretrained(model_version, output_attentions=True)
        tokenizer = RobertaTokenizer.from_pretrained(model_version)
        model = model.to(device)
        model = model.eval()
        print("--- Model loaded. ---")
    
    return model, tokenizer
    

# save_directory = "model_path"

# Load the model and tokenizer from the saved directory
# model = RobertaModel.from_pretrained(save_directory, output_attentions=True)
# tokenizer = RobertaTokenizer.from_pretrained(save_directory)
def ChemBERTa(input_file: str, output_file: str):

    # load the compound smiles
    smilesdf = pd.read_csv(input_file)
    smiles = smilesdf["SMILES"].tolist()

    smiles[0:4]

    len(smiles)

    # get the ChemBERTa embeddings
    final_df = pd.DataFrame()
    finalSMILES = [] # list of smiles for which embeddings were calculated successfully
    
    model, tokenizer = get_model()
    
    for smi in smiles:
    # print(smi)

    # Tokenize the smiles and obtain the tokens:
        encoded_input = tokenizer(smi, add_special_tokens=True, return_tensors='pt')
        encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
        
        # generate the embeddings
        with torch.no_grad():
            try:
                # this line gives error when smile size is greater than max_seq_length of 512, so enclosing the code in try
                model_output = model(**encoded_input)
            except:
                # print(smi) # will print the smiles that exceed the size limitation
                continue
            else:
                finalSMILES.append(smi)
            embeddings = model_output.last_hidden_state.mean(dim=1).cpu()
        
            # convert the emeddings output to a dataframe
            df = pd.DataFrame(embeddings).astype("float")
            final_df = pd.concat([final_df, df])


    # add a prefix to all the column names
    final_df = final_df.add_prefix('ChB77MLM_')
    final_df

    # for how many smiles the embeddings were successfully calculated
    print("Total SMILES: " + str(len(smiles)))
    print("SMILES converted: " + str(len(finalSMILES)))
    print("SMILES not converted: " + str(len(smiles) - len(finalSMILES)))


    # Subset the rows of df1 that have values of column A present in my_list
    df_subset = smilesdf.loc[smilesdf['SMILES'].isin(finalSMILES), ['Ligand_ID', 'SMILES']]
    df_subset

    final_df = pd.concat([df_subset.reset_index(drop=True), final_df.reset_index(drop=True)], axis=1)
    final_df

    # save the final as a csv
    final_df.to_csv(output_file, index=False)

    print("Code ran successfully")
    
    
parser = argparse.ArgumentParser(description="ChemBERTa")
parser.add_argument("--input", help="Input File Path")
parser.add_argument("--output", help="Output File Path", default="Raw_ChemBERTa.csv")

args = parser.parse_args()

ChemBERTa(args.input, args.output)