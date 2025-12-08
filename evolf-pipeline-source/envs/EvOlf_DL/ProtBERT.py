import torch
from transformers import BertModel, BertTokenizer
import re
import os
import requests
import numpy as np
import pandas as pd
import argparse
import random

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Using device: {device} ---")

model = None
tokenizer = None

# Lazy Loading
def get_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        # Load the vocabulary and ProtBert-BFD Model
        tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
        model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        model = model.to(device)
        model = model.eval()
    return model, tokenizer


def ProtBERT(input_file: str, output_file: str):
    
    model, tokenizer = get_model()
    
    RawData = pd.read_csv(input_file)

    # Filter out sequences with length >= 1024
    filtered_data = RawData[RawData["Sequence"].str.len() < 1024].reset_index(drop=True)

    # subset the receptor sequences
    sequences = filtered_data["Sequence"].tolist()


    # add a space after each amino acid of each sequence
    seq_split = list()
    for i in sequences:
        a = ' '.join(list(i))
        seq_split.append(a)



    ids = tokenizer.batch_encode_plus(seq_split, add_special_tokens=True, padding='longest')
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)


    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]


    embedding = embedding.cpu().numpy()


    features = [] 
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][1:seq_len-1]
        features.append(seq_emd)


    final_df = pd.DataFrame()

    for i in range(len(features)): # for each sequence
        # take means of all the aminoacids for all 1024 columns (colMeans) in the 2d array
        a = np.mean(features[i], axis=0)
        # convert it to a dataframe
        df = pd.DataFrame(a)
        # transpose the embeddind dataframe
        df = df.T
        # rename the columns to BFD_0, BFD_1...
        df = df.add_prefix('BFD_')
        # concat (rjoin) the results
        final_df = pd.concat([final_df, df], axis=0)


    # Reset the index of final_df
    final_df.reset_index(drop=True, inplace=True)

    # add information about the Receptor IDs
    final_df = pd.concat([filtered_data['Receptor_ID'], final_df], axis=1)

    final_df.to_csv(output_file, index=False)

    print("Code ran successfully")

parser = argparse.ArgumentParser(description="ProtBERT")
parser.add_argument("--input", help="Input File Path")
parser.add_argument("--output", help="Output File Path", default="Raw_ProtBERT.csv")

args = parser.parse_args()

ProtBERT(args.input, args.output)