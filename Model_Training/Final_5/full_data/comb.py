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


# Get the local time
st = time.time()
print(f'Code start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

"""
Loading all the required datasets based on the model
"""
# %%
file_path = f'../../../../all_data/processedData/MainModel/Full_data/full_data.pkl'
with open(file_path, 'rb') as file:
    dataTrain = pickle.load(file)

train_loader = DataLoader(dataTrain,  batch_size = batch_size, shuffle=True, worker_init_fn=worker_init_fn) 

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

weights_file_path = f"../weights"
train_text_file_path = f'../text/'

train_key_embedding_file_path = f'../embeddings/train/ligand/'
train_lock_embedding_file_path = f'../embeddings/train/receptor/'
train_concat_embedding_file_path = f'../embeddings/train/super_embed/'

test_key_embedding_file_path = f'../embeddings/test/ligand/'
test_lock_embedding_file_path = f'../embeddings/test/receptor/'
test_concat_embedding_file_path = f'../embeddings/test/super_embed/'


model.train()
model, train_list, loss_train, acc_train = trainModel(model, train_loader,
            weights_file_path, train_text_file_path, 
            train_key_embedding_file_path, train_lock_embedding_file_path, train_concat_embedding_file_path)



# %%
# print("Final Weights: ")
# for name, param in model.named_parameters():
#         if param.requires_grad:
#             print(name, param.data)


# %%
# print("comb best epoch:", best_epoch)
# print("comb best val acc:", best_val_acc)

# %%

"""
Making plots
"""
plt.plot(loss_train, label = "Train Loss",color='red')
plt.title("Loss Curve")
plt.savefig(f'../figures/Loss Curve.pdf', format="pdf", bbox_inches="tight") 
plt.legend()
plt.show()

# %%
plt.plot(acc_train, label = "Train Acc",color='red')
plt.title("Accuracy Curve")
plt.savefig(f'../figures/Accuracy Curve.pdf', format="pdf", bbox_inches="tight") 
plt.legend()
plt.show()

# %%

"""
First load the model with best weights
Performing testing on whatever dataset as neededed
"""

# best_epoch = 99

# model.load_state_dict(torch.load(f"{weights_file_path}/epoch_{best_epoch}.pt"))
# model.eval()

# _, _, _ = testModel(model, comb_test_loader, test_text_file_path, test_key_embedding_file_path, test_lock_embedding_file_path, test_concat_embedding_file_path, 'comb_test')

print(f'Code end time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))