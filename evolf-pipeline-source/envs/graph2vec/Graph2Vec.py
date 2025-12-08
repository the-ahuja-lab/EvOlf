# import libraries
from rdkit import Chem
import networkx as nx
from karateclub import Graph2Vec
import pandas as pd
import pickle
import argparse

CommonData = None

def load_resources():
    global CommonData
    if CommonData is None:
        CommonData = pd.read_csv("/app/models/Common_Input_G2V_15.csv")
        CommonData = CommonData.iloc[:, : 2]
    return CommonData

def mol_to_nx(mol):
    G = nx.Graph()

    if mol is None:  # Handle invalid molecules
        return G  # Return an empty graph or handle it as needed

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol())

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())

    return G


def Graph2Vec_func(input_file: str, output_file):
    
    load_resources()
    
    # Load the Data
    RawData = pd.read_csv(input_file)

    chem_df_c = pd.DataFrame(CommonData['SMILES'])
    # Convert SMILES into a molecule sturcture
    chem_df_c['mol'] = chem_df_c['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
    # Convert molecule sturctures into a graph
    chem_df_c['graph'] = chem_df_c['mol'].apply(lambda x: mol_to_nx(x))

    st=[]
    for i in range(len(RawData)):
        # print(i)
        RawData_sub = RawData.iloc[i,:]
        RawData_sub = pd.DataFrame(RawData_sub).T
        # print(RawData_sub['SMILES'])
        chem_df = pd.DataFrame(RawData_sub['SMILES'])
        # Convert SMILES into a molecule sturcture
        chem_df['mol'] = chem_df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
        # Convert molecule sturctures into a graph
        chem_df['graph'] = chem_df['mol'].apply(lambda x: mol_to_nx(x))
        # concat with the common molecules
        a = pd.concat([chem_df.reset_index(drop=True), chem_df_c.reset_index(drop=True)], axis = 0)
        a.reset_index(inplace=True, drop=True)

        model = Graph2Vec()
        model.fit(a['graph'])
        chem_graph2vec = model.get_embedding()
        chem_graph2vec = pd.DataFrame(chem_graph2vec)
        # rename all the columns
        chem_graph2vec = chem_graph2vec.add_prefix('Graph2Vec_')
        # drop all rows except the first one
        chem_graph2vec = chem_graph2vec.drop(chem_graph2vec.index.to_list()[1:], axis = 0)

        # Add information about the Ligand IDs and SMILES
        chem_graph2vec = pd.concat([RawData_sub[['Ligand_ID', 'SMILES']].reset_index(), chem_graph2vec], axis = 1)
        st.append(chem_graph2vec.drop('index', axis = 1))

        del model,chem_df,chem_graph2vec, a, RawData_sub

    res = pd.concat(st, axis=0, ignore_index=True)


    res.to_csv(output_file, index = False)

    print("Code ran successfully")

parser = argparse.ArgumentParser(description="Graph2Vec")
parser.add_argument("--input", help="Input File Path")
parser.add_argument("--output", help="Output File Path", default="Raw_Graph2Vec.csv")

args = parser.parse_args()

Graph2Vec_func(args.input, args.output)