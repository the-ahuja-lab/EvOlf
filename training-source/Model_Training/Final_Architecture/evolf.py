from architecture import *             # Contains the model architecture
from myDataset import *            # Necessary to define the dataset
from myImports import *                # Contains all import files
from myTrainParams import *            # Contains the code for training and testing a model


seed_value = 42
rs = RandomState(MT19937(SeedSequence(seed_value))) 
np.random.seed(seed_value)
batch_size = 64

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

# Define the folder names for all the files
dataType = "evolf" # comb, evolf, glass
dataSubType = "all" # all, human
outputDir = f'{dataType}_{dataSubType}'
inputPath = f'../../../../all_data/processedData/MainModel/SingleDataset'

# Get the local time
st = time.time()
print(f'Code start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

"""
Loading all the required datasets based on the model
"""
# %%
file_path = f'{inputPath}/{dataType}/{dataSubType}/train.pkl'
with open(file_path, 'rb') as file:
    dataTrain = pickle.load(file)
    
file_path = f'{inputPath}/{dataType}/{dataSubType}/{outputDir}_val.pkl'
with open(file_path, 'rb') as file:
    dataVal = pickle.load(file)

file_path = f'{inputPath}/{dataType}/{dataSubType}/comb_all_test.pkl'
with open(file_path, 'rb') as file:
    dataTest_comb_all = pickle.load(file)

file_path = f'{inputPath}/{dataType}/{dataSubType}/comb_human_test.pkl'
with open(file_path, 'rb') as file:
    dataTest_comb_human = pickle.load(file)

file_path = f'{inputPath}/{dataType}/{dataSubType}/evolf_all_test.pkl'
with open(file_path, 'rb') as file:
    dataTest_evolf_all = pickle.load(file)

file_path = f'{inputPath}/{dataType}/{dataSubType}/evolf_human_test.pkl'
with open(file_path, 'rb') as file:
    dataTest_evolf_human = pickle.load(file)

file_path = f'{inputPath}/{dataType}/{dataSubType}/glass_all_test.pkl'
with open(file_path, 'rb') as file:
    dataTest_glass_all = pickle.load(file)

file_path = f'{inputPath}/{dataType}/{dataSubType}/glass_human_test.pkl'
with open(file_path, 'rb') as file:
    dataTest_glass_human = pickle.load(file)


train_loader =  DataLoader(dataTrain,   batch_size = batch_size, shuffle=True, worker_init_fn=worker_init_fn) 
val_loader =    DataLoader(dataVal,     batch_size = batch_size, shuffle=True, worker_init_fn=worker_init_fn) 

test_loader_comb_all =      DataLoader(dataTest_comb_all,    batch_size = batch_size, shuffle=False, worker_init_fn=worker_init_fn)
test_loader_comb_human =    DataLoader(dataTest_comb_human,  batch_size = batch_size, shuffle=False, worker_init_fn=worker_init_fn)
test_loader_evolf_all =     DataLoader(dataTest_evolf_all,   batch_size = batch_size, shuffle=False, worker_init_fn=worker_init_fn)
test_loader_evolf_human =   DataLoader(dataTest_evolf_human, batch_size = batch_size, shuffle=False, worker_init_fn=worker_init_fn)
test_loader_glass_all =     DataLoader(dataTest_glass_all,   batch_size = batch_size, shuffle=False, worker_init_fn=worker_init_fn)
test_loader_glass_human =   DataLoader(dataTest_glass_human, batch_size = batch_size, shuffle=False, worker_init_fn=worker_init_fn)


# data = next(iter(comb_test_loader))
# ids, inputs, labels = data
# print(ids)
# print(labels)
# print(len(inputs))


# %%
"""
Loading the model instance onto GPU
"""
model = MyModel().to(device)

# # %%
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total trainable parameters: {total_params}")


# print("Initial weights:")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

# %%                  

"""
Setting all file paths to save information
"""                                          

weights_file_path = f"../weights/{outputDir}"
train_text_file_path = f'../text/{outputDir}/train/'
val_text_file_path = f'../text/{outputDir}/val/'
test_text_file_path = f'../text/{outputDir}/test/'

train_key_embedding_file_path = f'../embeddings/ligand/{outputDir}/train/'
train_lock_embedding_file_path = f'../embeddings/receptor/{outputDir}/train/'
train_concat_embedding_file_path = f'../embeddings/super_embed/{outputDir}/train/'

val_key_embedding_file_path = f'../embeddings/ligand/{outputDir}/val/'
val_lock_embedding_file_path = f'../embeddings/receptor/{outputDir}/val/'
val_concat_embedding_file_path = f'../embeddings/super_embed/{outputDir}/val/'

test_key_embedding_file_path = f'../embeddings/ligand/{outputDir}/test/'
test_lock_embedding_file_path = f'../embeddings/receptor/{outputDir}/test/'
test_concat_embedding_file_path = f'../embeddings/super_embed/{outputDir}/test/'

model.train()
model, train_list, val_list, best_epoch, best_val_acc, loss_train, loss_val, acc_train, acc_val = trainModel(model, train_loader, val_loader, "",
            weights_file_path, train_text_file_path, val_text_file_path, test_text_file_path, 
            train_key_embedding_file_path, train_lock_embedding_file_path, train_concat_embedding_file_path, 
            val_key_embedding_file_path, val_lock_embedding_file_path, val_concat_embedding_file_path)



# # %%
# # print("Final Weights: ")
# # for name, param in model.named_parameters():
# #         if param.requires_grad:
# #             print(name, param.data)


# %%
print(f"{outputDir} best epoch:", best_epoch)
print(f"{outputDir} best val acc:", best_val_acc)

# %%

"""
Making plots
"""
plt.plot(loss_train, label = "Train Loss",color='red')
plt.plot(loss_val, label = "Val Loss",color='green')
plt.title("Loss Curve")
plt.savefig(f'../figures/{outputDir}/Loss Curve.pdf', format="pdf", bbox_inches="tight") 
plt.legend()
plt.show()

# %%
plt.plot(acc_train, label = "Train Acc",color='red')
plt.plot(acc_val, label = "Val Acc",color='green')
plt.title("Accuracy Curve")
plt.savefig(f'../figures/{outputDir}/Accuracy Curve.pdf', format="pdf", bbox_inches="tight") 
plt.legend()
plt.show()

# %%

"""
First load the model with best weights
Performing testing on whatever dataset as neededed
"""
# best_epoch = 1
model.load_state_dict(torch.load(f"{weights_file_path}/epoch_{best_epoch}.pt"))
model.eval()


_, _, _ = testModel(model, test_loader_comb_all,    test_text_file_path, test_key_embedding_file_path, test_lock_embedding_file_path, test_concat_embedding_file_path, 'comb_all_test')
_, _, _ = testModel(model, test_loader_comb_human,  test_text_file_path, test_key_embedding_file_path, test_lock_embedding_file_path, test_concat_embedding_file_path, 'comb_human_test')
_, _, _ = testModel(model, test_loader_evolf_all,   test_text_file_path, test_key_embedding_file_path, test_lock_embedding_file_path, test_concat_embedding_file_path, 'evolf_all_test')
_, _, _ = testModel(model, test_loader_evolf_human, test_text_file_path, test_key_embedding_file_path, test_lock_embedding_file_path, test_concat_embedding_file_path, 'evolf_human_test')
_, _, _ = testModel(model, test_loader_glass_all,   test_text_file_path, test_key_embedding_file_path, test_lock_embedding_file_path, test_concat_embedding_file_path, 'glass_all_test')
_, _, _ = testModel(model, test_loader_glass_human, test_text_file_path, test_key_embedding_file_path, test_lock_embedding_file_path, test_concat_embedding_file_path, 'glass_human_test')


# Calibration Plot
df = pd.read_csv(f'{test_text_file_path}{outputDir}_test.csv')

y_probs = df['P1'].values
y_test = df['Actual Label'].values

prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)

# Plotting the calibration curve
plt.plot(prob_pred, prob_true, marker='o', linestyle='-', color='b')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line for reference
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plot')
plt.savefig(f'../figures/{outputDir}/Calibration Curve.pdf', format="pdf", bbox_inches="tight") 
plt.show()


print(f'Code end time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))