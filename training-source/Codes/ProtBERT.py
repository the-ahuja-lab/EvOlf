import torch
from transformers import BertModel, BertTokenizer
import re
import os
import requests
from tqdm.auto import tqdm
import numpy as np
import pandas as pd


device = torch.device('cpu')
print("Using {}".format(device))


# Load the vocabulary and ProtBert-BFD Model
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")


model = model.to(device)
model = model.eval()


dataPath = "/storage1/aayushim/EvOlf_3.3/01_Dry_Lab/01_Data/16_ArunShukla_GPCRs/"
outPath = "/storage1/aayushim/EvOlf_3.3/01_Dry_Lab/02_Descriptors/31_ASGPCRs/"


RawData = pd.read_csv(dataPath+"recsData.csv")
RawData

# Filter out sequences with length >= 1024
filtered_data = RawData[RawData["Sequence"].str.len() < 1024].reset_index(drop=True)
filtered_data

# subset the receptor sequences
sequences = filtered_data["Sequence"].tolist()
# sequences[:5]


# add a space after each amino acid of each sequence
seq_split = list()
for i in sequences:
  a = ' '.join(list(i))
  seq_split.append(a)

# seq_split[:5]


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


print(len(features)) # total sequences
print(len(features[0])) # total aa in 1st sequence
print(len(features[0][0])) # 1024 embeddings for each aminoacid


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
final_df

final_df.to_csv(outPath+'Raw_ProtBERT.csv', index=False)

print("Code ran successfully")