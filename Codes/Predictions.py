import sys
sys.path.append('/content/my_modules')

from architecture import *             # Contains the model architecture
from myDataset import *                # Necessary to define the dataset
from myImports import *                # Contains all import files
from myTrainParams import *            # Contains the code for training and testing a model

# Get the time
st = time.time()
print(f'Code start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

# define the file paths for saving and loading
weights_file_path = f"../../Final_5" ##MODEL PATH FROM MODEL SUBFOLDER
test_text_file_path = f'../text/' ##TRIAL OUTPUT TEXT FOLDER
test_key_embedding_file_path = f'../embeddings/ligand/' ##TRIAL OUTPUT
test_lock_embedding_file_path = f'../embeddings/receptor/' ## TRIAL OUTPUT
test_concat_embedding_file_path = f'../embeddings/super_embed/' ## TRIAL OUTPUT

# define the input file path
mypath = "../Sample_data/" #SAMPLE DATA
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# load the ligand feature files
ligs_g2v = pd.read_csv(mypath+"Graph2Vec_Final.csv")
ligs_sig = pd.read_csv(mypath+"Signaturizer_Final.csv")
ligs_mord = pd.read_csv(mypath+"Mordred_Final.csv")
ligs_m2v = pd.read_csv(mypath+"Mol2Vec_Final.csv")
ligs_cb = pd.read_csv(mypath+"ChemBERTa_Final.csv")

# load the receptor feature files
recs_bfd = pd.read_csv(mypath+"ProtBERT_Final.csv")
recs_t5 = pd.read_csv(mypath+"ProtT5_Final.csv")
recs_mf = pd.read_csv(mypath+"MathFeature_Final.csv")
recs_r = pd.read_csv(mypath+"ProtR_Final.csv")

print(f'Data Loaded:', time.strftime("%H:%M:%S", time.gmtime(time.time() - st)))


best_epoch = 100
model = MyModel().to(device)
model.load_state_dict(torch.load(f"{weights_file_path}/epoch_{best_epoch}.pt", map_location=torch.device('cpu')))
model.eval()

# load the main data files
main_data = pd.read_csv(mypath+"mainData.csv")
main_data = main_data[["IDs","Ligand_ID","Receptor_ID"]]

# Create a mapping from main data to define the sort order
sort_order = pd.Series(main_data.index, index=main_data['IDs']).to_dict()

# merge the features with the main data
Key_1 = pd.merge(main_data, ligs_g2v, on='Ligand_ID', how='inner')
Key_2 = pd.merge(main_data, ligs_sig, on='Ligand_ID', how='inner')
Key_3 = pd.merge(main_data, ligs_mord, on='Ligand_ID', how='inner')
Key_4 = pd.merge(main_data, ligs_m2v, on='Ligand_ID', how='inner')
Key_5 = pd.merge(main_data, ligs_cb, on='Ligand_ID', how='inner')

Lock_1 = pd.merge(main_data, recs_bfd, on='Receptor_ID', how='inner')
Lock_2 = pd.merge(main_data, recs_t5, on='Receptor_ID', how='inner')
Lock_3 = pd.merge(main_data, recs_mf, on='Receptor_ID', how='inner')
Lock_4 = pd.merge(main_data, recs_r, on='Receptor_ID', how='inner')

# map the sorting order based on main data
Key_1['sort_order'] = Key_1['IDs'].map(sort_order)
Key_2['sort_order'] = Key_2['IDs'].map(sort_order)
Key_3['sort_order'] = Key_3['IDs'].map(sort_order)
Key_4['sort_order'] = Key_4['IDs'].map(sort_order)
Key_5['sort_order'] = Key_5['IDs'].map(sort_order)

Lock_1['sort_order'] = Lock_1['IDs'].map(sort_order)
Lock_2['sort_order'] = Lock_2['IDs'].map(sort_order)
Lock_3['sort_order'] = Lock_3['IDs'].map(sort_order)
Lock_4['sort_order'] = Lock_4['IDs'].map(sort_order)

# sort the dataframe -> drop the sort column -> reset the index
Key_1.sort_values(by='sort_order', inplace=True)
Key_1.drop(columns=['sort_order'], inplace=True)
Key_1.reset_index(drop=True, inplace=True)

Key_2.sort_values(by='sort_order', inplace=True)
Key_2.drop(columns=['sort_order'], inplace=True)
Key_2.reset_index(drop=True, inplace=True)

Key_3.sort_values(by='sort_order', inplace=True)
Key_3.drop(columns=['sort_order'], inplace=True)
Key_3.reset_index(drop=True, inplace=True)

Key_4.sort_values(by='sort_order', inplace=True)
Key_4.drop(columns=['sort_order'], inplace=True)
Key_4.reset_index(drop=True, inplace=True)

Key_5.sort_values(by='sort_order', inplace=True)
Key_5.drop(columns=['sort_order'], inplace=True)
Key_5.reset_index(drop=True, inplace=True)


Lock_1.sort_values(by='sort_order', inplace=True)
Lock_1.drop(columns=['sort_order'], inplace=True)
Lock_1.reset_index(drop=True, inplace=True)

Lock_2.sort_values(by='sort_order', inplace=True)
Lock_2.drop(columns=['sort_order'], inplace=True)
Lock_2.reset_index(drop=True, inplace=True)

Lock_3.sort_values(by='sort_order', inplace=True)
Lock_3.drop(columns=['sort_order'], inplace=True)
Lock_3.reset_index(drop=True, inplace=True)

Lock_4.sort_values(by='sort_order', inplace=True)
Lock_4.drop(columns=['sort_order'], inplace=True)
Lock_4.reset_index(drop=True, inplace=True)

ids = main_data['IDs']

Key_1 =  np.array(Key_1.iloc[:, 4:])
Key_2 =  np.array(Key_2.iloc[:, 4:])
Key_3 =  np.array(Key_3.iloc[:, 4:])
Key_4 =  np.array(Key_4.iloc[:, 4:])
Key_5 =  np.array(Key_5.iloc[:, 4:])

Lock_1 = np.array(Lock_1.iloc[:, 3:])
Lock_2 = np.array(Lock_2.iloc[:, 3:])
Lock_3 = np.array(Lock_3.iloc[:, 3:])
Lock_4 = np.array(Lock_4.iloc[:, 3:])

print(f'Time elapsed till Data Intersection : {time.strftime("%H:%M:%S", time.gmtime(time.time() - st))}')

full_data = [Key_1, Key_2, Key_3, Key_4, Key_5, Lock_1, Lock_2, Lock_3, Lock_4]

file_path = '/storage/aayushim/mudit/EvOlf/Pipeline/all_data/processingModels/scaler_models/full_data/comb_all/' ## IN MODEL FOLDER


ind = 0
for i in range(1, 6):
  scaler = pickleRead(file_path, f'Ligand_{i}.pkl')
  full_data[ind] = scaler.transform(full_data[ind])
  ind += 1
  # print("Scaling Key ", ind ,"done")

for i in range(1, 5):
  scaler = pickleRead(file_path, f'Receptor_{i}.pkl')
  full_data[ind] = scaler.transform(full_data[ind])
  ind += 1
  # print("Scaling Lock ", ind ,"done")

print(f'Time elapsed till Scaling : {time.strftime("%H:%M:%S", time.gmtime(time.time() - st))}')

file_path = '/storage/aayushim/mudit/EvOlf/Pipeline/all_data/processingModels/pca_models/full_data/comb_all/' ## IN MODEL FOLDER

ind = 0
for i in range(1, 6):
  pca = pickleRead(file_path, f'Ligand_{i}.pkl')
  full_data[ind] = pca.transform(full_data[ind])
  ind += 1

for i in range(1, 5):
  pca = pickleRead(file_path, f'Receptor_{i}.pkl')
  full_data[ind] = pca.transform(full_data[ind])
  ind += 1

print(f'Time elapsed till PCA : {time.strftime("%H:%M:%S", time.gmtime(time.time() - st))}')

Key_1 = np.array(full_data[0], dtype = np.float32)
Key_2 = np.array(full_data[1], dtype = np.float32)
Key_3 = np.array(full_data[2], dtype = np.float32)
Key_4 = np.array(full_data[3], dtype = np.float32)
Key_5 = np.array(full_data[4], dtype = np.float32)

Lock_1 = np.array(full_data[5], dtype = np.float32)
Lock_2 = np.array(full_data[6], dtype = np.float32)
Lock_3 = np.array(full_data[7], dtype = np.float32)
Lock_4 = np.array(full_data[8], dtype = np.float32)

Key_1 = Key_1.reshape(Key_1.shape[0], 1, Key_1.shape[1])
Key_2 = Key_2.reshape(Key_2.shape[0], 1, Key_2.shape[1])
Key_3 = Key_3.reshape(Key_3.shape[0], 1, Key_3.shape[1])
Key_4 = Key_4.reshape(Key_4.shape[0], 1, Key_4.shape[1])
Key_5 = Key_5.reshape(Key_5.shape[0], 1, Key_5.shape[1])

Lock_1 = Lock_1.reshape(Lock_1.shape[0], 1, Lock_1.shape[1])
Lock_2 = Lock_2.reshape(Lock_2.shape[0], 1, Lock_2.shape[1])
Lock_3 = Lock_3.reshape(Lock_3.shape[0], 1, Lock_3.shape[1])
Lock_4 = Lock_4.reshape(Lock_4.shape[0], 1, Lock_4.shape[1])

Key_dict = {}

Key_dict[0] = Key_1
Key_dict[1] = Key_2
Key_dict[2] = Key_3
Key_dict[3] = Key_4
Key_dict[4] = Key_5

Lock_dict = {}

Lock_dict[0] = Lock_1
Lock_dict[1] = Lock_2
Lock_dict[2] = Lock_3
Lock_dict[3] = Lock_4

num_files_key = 5
num_files_Lock = 4

n = len(ids)
# n = len(ids)

test_X = []
test_y = []

# print(n, num_files_key)
# print(Key_dict)
# print(len(Key_dict.keys()))
# print(len(Key_dict.values()))

for i in range(n):
  lst = []
  for j in range(num_files_key):
    # print(j,i)
    # print(len(Key_dict))
    # print(len(Key_dict[j]))
    lst.append(Key_dict[j][i])
  for j in range(num_files_Lock):
    lst.append(Lock_dict[j][i])
  my_list = [ids[i], lst]
  test_y.append(-1)
  test_X.append(my_list)


test_data = LigandReceptorDataset(np.array(test_X)[:, 0].tolist(), np.array(test_X)[:, 1].tolist(), test_y)  
test_loader = DataLoader(test_data,  batch_size = batch_size, shuffle=False, worker_init_fn=worker_init_fn) 

_, _ = testModel(model, test_loader, test_text_file_path, test_key_embedding_file_path, test_lock_embedding_file_path, test_concat_embedding_file_path, f'Prediction_Output')

print(f'Code end time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

print("Code ran successfully")