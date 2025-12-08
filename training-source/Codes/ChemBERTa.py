from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer
import torch
import pandas as pd

# download the pretrained model
model_version = 'DeepChem/ChemBERTa-77M-MLM'

# download and load the tokenizer which is used for pretraining the above model
model = RobertaModel.from_pretrained(model_version, output_attentions=True)
tokenizer = RobertaTokenizer.from_pretrained(model_version)

# save_directory = "model_path"

# Load the model and tokenizer from the saved directory
# model = RobertaModel.from_pretrained(save_directory, output_attentions=True)
# tokenizer = RobertaTokenizer.from_pretrained(save_directory)

outPath = "/content/drive/MyDrive/PhD/Projects/eNOSE/EvOlf_3.3/15_Colab/Sample_Data/"
dataPath = "/content/drive/MyDrive/PhD/Projects/eNOSE/EvOlf_3.3/15_Colab/Trial_Run_Outputs/"

# load the compound smiles
smilesdf = pd.read_csv(dataPath+"ligsData.csv")
smiles = smilesdf["SMILES"].tolist()

smiles[0:4]

len(smiles)

# get the ChemBERTa embeddings
final_df = pd.DataFrame()
finalSMILES = [] # list of smiles for which embeddings were calculated successfully

for smi in smiles:
  # print(smi)

  # Tokenize the smiles and obtain the tokens:
  encoded_input = tokenizer(smi, add_special_tokens=True, return_tensors='pt')
  
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
    embeddings = model_output.last_hidden_state.mean(dim=1)
  
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
final_df.to_csv(outPath+'Raw_ChemBERTa.csv', index=False)

print("Code ran successfully")