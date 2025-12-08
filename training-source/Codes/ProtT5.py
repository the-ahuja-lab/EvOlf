# Set working directory
import os
#os.chdir('/storage1/aayushim/EvOlf_3.3/01_Dry_Lab/02_Descriptors/31_ASGPCRs')

# In the following you can define your desired output. Current options:
# per_residue embeddings
# per_protein embeddingsos

# Replace this file with your own (multi-)FASTA
# Headers are expected to start with ">";
seq_path = "/storage1/aayushim/EvOlf_3.3/01_Dry_Lab/02_Descriptors/31_ASGPCRs/02_ASGPCRs.fasta"

# whether to retrieve embeddings for each residue in a protein 
# --> Lx1024 matrix per protein with L being the protein's length
# as a rule of thumb: 1k proteins require around 1GB RAM/disk
per_residue = True 
#per_residue_path = "/storage1/aayushim/EvOlf_3.3/01_Dry_Lab/02_Descriptors/protT5/output/per_residue_embeddings_ASGPCRs.h5" # where to store the embeddings

# whether to retrieve per-protein embeddings 
# --> only one 1024-d vector per protein, irrespective of its length
per_protein = True
per_protein_path = "/storage1/aayushim/EvOlf_3.3/01_Dry_Lab/02_Descriptors/protT5/output/per_protein_embeddings_ASGPCRs.h5" # where to store the embeddings

# make sure that either per-residue or per-protein embeddings are stored
assert per_protein is True or per_residue is True or sec_struct is True, print(
    "Minimally, you need to active per_residue, per_protein or sec_struct. (or any combination)")


# Import dependencies and check whether GPU is available.

#@title Import dependencies and check whether GPU is available. { display-mode: "form" }
from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import time
import pandas as pd
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print("Using {}".format(device))



# Load encoder-part of ProtT5 in half-precision.

#@title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50) 
def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer



# Read in file in fasta format.

#@title Read in file in fasta format. { display-mode: "form" }
def read_fasta( fasta_path, split_char="!", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    seqs = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                seqs[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq= ''.join( line.split() ).upper().replace("-","")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U','X').replace('Z','X').replace('O','X')
                seqs[ uniprot_id ] += seq 
    example_id=next(iter(seqs))
    print("Read {} sequences.".format(len(seqs)))
    # print("Example:\n{}\n{}".format(example_id,seqs[example_id]))

    return seqs


#@title Generate embeddings. { display-mode: "form" }
# Generate embeddings via batch-processing
# per_residue indicates that embeddings for each residue in a protein should be returned.
# per_protein indicates that embeddings for a whole protein should be returned (average-pooling)
# max_residues gives the upper limit of residues within one batch
# max_seq_len gives the upper sequences length for applying batch-processing
# max_batch gives the upper number of sequences per batch
def get_embeddings( model, tokenizer, seqs, per_residue, per_protein,
                   max_residues=4000, max_seq_len=1000, max_batch=100 ):

    results = {"residue_embs" : dict(), 
               "protein_embs" : dict()
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
            
            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                # print(identifier)
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim  
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len] # max pooling
                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
                if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()


    passed_time=time.time()-start
    avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')
    return results


# Write embeddings to disk

#@title Write embeddings to disk. { display-mode: "form" }
def save_embeddings(emb_dict,out_path):
    with h5py.File(str(out_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)
    return None



# Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
model, tokenizer = get_T5_model()

# Load example fasta.
seqs = read_fasta( seq_path )

# Compute embeddings and/or secondary structure predictions
results = get_embeddings(model, tokenizer, seqs,
                         per_residue, per_protein)

# Store per-residue embeddings
#if per_residue:
#  save_embeddings(results["residue_embs"], per_residue_path)
if per_protein:
  save_embeddings(results["protein_embs"], per_protein_path)


f = h5py.File('/storage1/aayushim/EvOlf_3.3/01_Dry_Lab/02_Descriptors/protT5/output/per_protein_embeddings_ASGPCRs.h5', 'r')

group_keys = list(f.keys())

final_df = pd.DataFrame()

for group_key in group_keys:
  # print('######', group_key, '######')
  # get the embedding values for each sequence
  data = f[group_key]
  # convert it to a dataframe
  df = pd.DataFrame(data)
  # transpose the embeddind dataframe
  dfT = df.T
  # rename the columns to PT5_0, PT5_1...
  dfT = dfT.add_prefix('PT5_')
  # rename the row name to the fasta header
  dfT.rename(index={0: group_key}, inplace=True)
  # concat (rjoin) the results
  final_df = pd.concat([final_df, dfT], axis=0)


# rename the index column - for csv
final_df.index.names = ['Receptor_ID']

# insert the receptor ids as a first column also - for rds
final_df.insert(loc = 0, column='Receptor_ID', value = final_df.index)

# save the final to csv
final_df.to_csv('Raw_ProtT5.csv', index = False)
# pyreadr.write_rds('08_EvOlf_ProtT5.rds', Signatures, compress = "gzip")

# Close the file to prevent corruption of the file
f.close()

print("Code ran successfully")