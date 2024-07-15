from architecture import *             # Contains the model architecture
from myDataset import *                # Necessary to define the dataset
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
dataType = "comb" # comb, evolf, glass
dataSubType = "all" # all, human
outputDir = f'{dataType}_{dataSubType}'

"""
Loading all the required datasets based on the model
"""
file_path = f'../train.pkl'
with open(file_path, 'rb') as file:
    dataTrain = pickle.load(file)
    
file_path = f'../comb_all_val.pkl'
with open(file_path, 'rb') as file:
    dataVal = pickle.load(file)

file_path = f'../comb_all_test.pkl'
with open(file_path, 'rb') as file:
    dataTest_comb_all = pickle.load(file)


train_loader =  DataLoader(dataTrain, batch_size = batch_size, shuffle=True, worker_init_fn=worker_init_fn) 
val_loader = DataLoader(dataVal, batch_size = batch_size, shuffle=True, worker_init_fn=worker_init_fn) 
test_loader_comb_all =  DataLoader(dataTest_comb_all, batch_size = batch_size, shuffle=False, worker_init_fn=worker_init_fn)


"""
Loading the model instance onto GPU
"""
model = MyModel().to(device)
                

"""
Setting all file paths to save information
"""                                          

weights_file_path = f"../weights/"
train_text_file_path = f'../text/train/'
val_text_file_path = f'../text/val/'
test_text_file_path = f'../text/test/'

train_key_embedding_file_path = f'../embeddings/ligand/train/'
train_lock_embedding_file_path = f'../embeddings/receptor/train/'
train_concat_embedding_file_path = f'../embeddings/super_embed/train/'

val_key_embedding_file_path = f'../embeddings/ligand/val/'
val_lock_embedding_file_path = f'../embeddings/receptor/val/'
val_concat_embedding_file_path = f'../embeddings/super_embed/val/'

test_key_embedding_file_path = f'../embeddings/ligand/test/'
test_lock_embedding_file_path = f'../embeddings/receptor/test/'
test_concat_embedding_file_path = f'../embeddings/super_embed/test/'

model.train()
model, train_list, val_list, best_epoch, best_val_acc, loss_train, loss_val, acc_train, acc_val = trainModel(model, train_loader, val_loader, "",
            weights_file_path, train_text_file_path, val_text_file_path, test_text_file_path, 
            train_key_embedding_file_path, train_lock_embedding_file_path, train_concat_embedding_file_path, 
            val_key_embedding_file_path, val_lock_embedding_file_path, val_concat_embedding_file_path)


print(f"{outputDir} best epoch:", best_epoch)
print(f"{outputDir} best val acc:", best_val_acc)


"""
Making plots
"""
plt.plot(loss_train, label = "Train Loss",color='red')
plt.plot(loss_val, label = "Val Loss",color='green')
plt.title("Loss Curve")
plt.savefig(f'../figures/Loss Curve.pdf', format="pdf", bbox_inches="tight") 
plt.legend()
plt.show()

plt.plot(acc_train, label = "Train Acc",color='red')
plt.plot(acc_val, label = "Val Acc",color='green')
plt.title("Accuracy Curve")
plt.savefig(f'../figures/Accuracy Curve.pdf', format="pdf", bbox_inches="tight") 
plt.legend()
plt.show()


"""
First, load the model with the best weights
Performing testing on whatever dataset as needed
"""

model.load_state_dict(torch.load(f"{weights_file_path}/epoch_{best_epoch}.pt"))
model.eval()


_, _, _ = testModel(model, test_loader_comb_all, test_text_file_path, test_key_embedding_file_path, test_lock_embedding_file_path, test_concat_embedding_file_path, 'comb_all_test')
