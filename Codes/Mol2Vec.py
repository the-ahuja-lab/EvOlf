import numpy as np
import pandas as pd
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, IdentifierTable, mol_to_svg
from gensim.models import word2vec

# The Mol2Vec package is not updated to account for updated Gensim. This function accounts for it.

def sentences2vecMe(sentences, model, unseen=None):
    """Generate vectors for each sentence (list) in a list of sentences. Vector is simply a
    sum of vectors for individual words.
    
    Parameters
    ----------
    sentences : list, array
        List with sentences
    model : word2vec.Word2Vec
        Gensim word2vec model
    unseen : None, str
        Keyword for unseen words. If None, those words are skipped.
        https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032

    Returns
    -------
    np.array
    """
    
    keys = set(model.wv.key_to_index)
    vec = []
    
    if unseen:
        unseen_vec = model.wv.get_vector(unseen)

    for sentence in sentences:
        if unseen:
            vec.append(sum([model.wv.get_vector(y) if y in set(sentence) & keys
                       else unseen_vec for y in sentence]))
        else:
            vec.append(sum([model.wv.get_vector(y) for y in sentence 
                            if y in set(sentence) & keys]))
    return np.array(vec)


# load the model
model = word2vec.Word2Vec.load('/storage1/aayushim/EvOlf_3.3/01_Dry_Lab/02_Descriptors/Mol2Vec/model_300dim.pkl')


dataPath = "/storage1/aayushim/EvOlf_3.3/01_Dry_Lab/01_Data/18_YMDB/"
outPath = "/storage1/aayushim/EvOlf_3.3/01_Dry_Lab/02_Descriptors/33_YMDB/"


RawData = pd.read_csv(dataPath + "ligsData.csv")
RawData


chem_df = pd.DataFrame(RawData[['Ligand_ID','SMILES']])
chem_df


# transforms smiles strings to mol rdkit object
chem_df['mol'] = chem_df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
chem_df



# Create a boolean mask to identify rows with None in column "A"
mask = chem_df['mol'].isnull()

# Get the rows where column "A" has None
rows_with_none = chem_df[mask]
idx = chem_df.index[mask]

# drop the molecules that were not converted
chem_df = chem_df.drop(index = idx)
chem_df[83:90]

# reset the index
chem_df.reset_index(inplace = True)



# convert the molecules into sentences, aka get the words (substructures) for a sentence (molecule)
chem_df['sentence'] = chem_df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)

# get the final suumed embeddings for each molecule 
chem_df['mol2vec'] = [DfVec(x) for x in sentences2vecMe(chem_df['sentence'], model, unseen='UNK')]


# convert the embeddings into numerical values
embeddings = np.array([x.vec for x in chem_df['mol2vec']])
embeddings


# convert to dataframe
embeddings_df = pd.DataFrame(embeddings)
# rename all the columns
embeddings_df = embeddings_df.add_prefix('Mol2Vec_')
embeddings_df


# Drop the molecules which were not converted from the RawData also
RawData = RawData.drop(index = idx)
# print(RawData[83:90])
# print(RawData.shape)

# check if the two dataframes have same smiles before resetting the index:
RawData.reset_index(inplace = True)
# print(sum(chem_df['SMILES'] == RawData['SMILES']))


# add smiles and ligand id
embeddings_df = pd.concat([RawData['Ligand_ID'], RawData['SMILES'], embeddings_df], axis=1)
embeddings_df


# check the data for missing values
sum(embeddings_df.isna().all(axis=1))


embeddings_df.to_csv(outPath+"Raw_Mol2Vec.csv", index = False)

print("Code ran successfully")