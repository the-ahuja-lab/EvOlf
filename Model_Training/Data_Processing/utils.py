from myImports import *

set_seed()

def loadFullData():
    return pickleRead('../all_data/processedData/FullData/', 'full_data.pkl')

def load_data(path = '../all_data/processedData/MainModel/SingleDataset/comb/all/0'):
    res = []
    for mode in ['train', 'val', 'test']:
        with open(f'{path}/{mode}.pkl', 'rb') as f:
            res += [pickle.load(f)]
    return res[0], res[1], res[2]

def accuracy(y_test, y_pred, verbose=False):
    m = y_test.shape[0]
    predicted = torch.max(y_pred, 1)[1]
    correct = (predicted == y_test).float().sum().item()
    if verbose: print('Accuracy:', correct,m)
    accuracy = correct/m
    return accuracy, correct, predicted


def Train(Net, train_data, val_data, batch_size, epochs, log_dir, model_name, store_predictions=False, store_embeddings=False, updateBatch=False, device='cpu', Loss=nn.CrossEntropyLoss(reduction='sum')):
    losses = []
    accs = []
    val_losses=[]
    val_accL=[]
    Net.to(device)
    for e in range(epochs):
        Net.train()
        step=0
        tot_loss=0.0
        start_time = time.time()
        correct_samples = 0
        total_samples = 0
        train_loader =DataLoader(train_data, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
        all_ids, all_y_pred, all_y_actual, all_y_p1 = np.array([]), np.array([]), np.array([]), np.array([])
        ligand_embeddings = None
        receptor_embeddings = None
        joint_embeddings = None
        for data in train_loader:
            ids, inputs, y = data
            lig1, lig2, lig3, lig4, lig5, rec1, rec2, rec3, rec4 = inputs
            lig1, lig2, lig3, lig4, lig5, rec1, rec2, rec3, rec4 = lig1.to(device), lig2.to(device), lig3.to(device), lig4.to(device), lig5.to(device), rec1.to(device), rec2.to(device), rec3.to(device), rec4.to(device)
            y = y.to(device)

            ligand = torch.cat([lig1, lig2, lig3, lig4, lig5], dim=1)
            receptor = torch.cat([rec1, rec2, rec3, rec4], dim=1)

            y_pred, ligand_embed, receptor_embed, joint_embed = Net(ligand, receptor)
            loss = Loss(y_pred,y)

            y_pred = nn.Softmax(dim = 1)(y_pred)

            Net.optimizer.zero_grad()
            loss.backward()
            Net.optimizer.step()

            step+=1
            tot_loss+=loss
            total_samples += y.shape[0]
            _, i_cor_sam, prediction = accuracy(y, y_pred, verbose=False)
            correct_samples += i_cor_sam

            if store_predictions == True:
                all_ids = np.concatenate((all_ids, np.array(ids)), axis = None)
                all_y_pred = np.concatenate((all_y_pred, prediction.cpu().detach().numpy()), axis = None)
                all_y_actual = np.concatenate((all_y_actual, y.cpu().detach().numpy()), axis = None)
                all_y_p1 = np.concatenate((all_y_p1, np.reshape(y_pred[:, :1].cpu().detach().numpy(), y_pred[:, :1].cpu().detach().numpy().shape[0])), axis = None)
            if store_embeddings == True:
                if ligand_embeddings is None:
                    ligand_embeddings = ligand_embed.cpu().detach().numpy()
                    receptor_embeddings = receptor_embed.cpu().detach().numpy()
                    joint_embeddings = joint_embed.cpu().detach().numpy()
                
                else:
                    ligand_embeddings = np.append(ligand_embeddings, ligand_embed.cpu().detach().numpy(), axis = None)
                    receptor_embeddings = np.append(receptor_embeddings, receptor_embed.cpu().detach().numpy(), axis = None)
                    joint_embeddings = np.append(joint_embeddings, joint_embed.cpu().detach().numpy(), axis = None)
        
        
        if store_predictions:
            writePredictionsIntoFile(f'{log_dir}/train_logs/{model_name}/predictions/', f'epoch_{e+1}.csv', all_ids, all_y_pred, all_y_actual, all_y_p1)
        if store_embeddings:
            writeEmbeddingsIntoFile(f'{log_dir}/train_logs/{model_name}/embeddings/', f'ligand_embeddings.csv', all_ids, ligand_embeddings)
            writeEmbeddingsIntoFile(f'{log_dir}/train_logs/{model_name}/embeddings/', f'receptor_embeddings.csv', all_ids, receptor_embeddings)
            writeEmbeddingsIntoFile(f'{log_dir}/train_logs/{model_name}/embeddings/', f'joint_embeddings.csv', all_ids, joint_embeddings)

        
        
        end_time = time.time()
        t = end_time-start_time
        l = tot_loss.item()/total_samples
        losses += [l]
        a = correct_samples/total_samples
        accs += [a]
        print('Epoch %2d Loss: %2.5e Accuracy: %2.5f Epoch Time: %2.5f' %(e,l,a,t))
        Net.scheduler.step(l)
        
        if val_data is not None:
            val_l, val_a = Test(Net, val_data, batch_size, 'validation', log_dir, model_name, store_predictions, store_embeddings, device)
            val_losses += [val_l]
            val_accL += [val_a]

        
        if updateBatch:
            if e % (epochs//4 + 1) == 0:
                batch_size += 64

        
    saveModel(Net, f"{log_dir}/saved_models", model_name)
    

    return Net, losses, accs, val_losses, val_accL


def Test(Net, test_data, batch_size, mode = 'test', log_dir='', model_name='', store_predictions=False, store_embeddings=False, device='cpu', Loss=nn.CrossEntropyLoss(reduction='sum')):
    Net.to(device)
    Net.eval()
    start_time = time.time()
    total_samples = 0
    correct_samples = 0
    loss = 0.0
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn)
    all_ids, all_y_pred, all_y_actual, all_y_p1 = np.array([]), np.array([]), np.array([]), np.array([])
    ligand_embeddings = None
    receptor_embeddings = None
    joint_embeddings = None

    for data in test_loader:
        ids, inputs, y = data
        lig1, lig2, lig3, lig4, lig5, rec1, rec2, rec3, rec4 = inputs
        lig1, lig2, lig3, lig4, lig5, rec1, rec2, rec3, rec4 = lig1.to(device), lig2.to(device), lig3.to(device), lig4.to(device), lig5.to(device), rec1.to(device), rec2.to(device), rec3.to(device), rec4.to(device)
        y = y.to(device)

        ligand = torch.cat([lig1, lig2, lig3, lig4, lig5], dim=1)
        receptor = torch.cat([rec1, rec2, rec3, rec4], dim=1)

        y_pred, ligand_embed, receptor_embed, joint_embed = Net(ligand, receptor)
        total_samples += y.shape[0]
        _, i_cor_sam, prediction = accuracy(y, y_pred, verbose=False)
        correct_samples += i_cor_sam
        loss += Loss(y_pred, y).cpu().detach().item()

        y_pred = nn.Softmax(dim = 1)(y_pred)
        all_y_pred = np.concatenate((all_y_pred, prediction.cpu().detach().numpy()), axis = None)
        all_y_actual = np.concatenate((all_y_actual, y.cpu().detach().numpy()), axis = None)
                
        if store_predictions == True:
                all_ids = np.concatenate((all_ids, np.array(ids)), axis = None)
                all_y_p1 = np.concatenate((all_y_p1, np.reshape(y_pred[:, :1].cpu().detach().numpy(), y_pred[:, :1].cpu().detach().numpy().shape[0])), axis = None)
        if store_embeddings == True:
            if ligand_embeddings is None:
                ligand_embeddings = ligand_embed.cpu().detach().numpy()
                receptor_embeddings = receptor_embed.cpu().detach().numpy()
                joint_embeddings = joint_embed.cpu().detach().numpy()
            else:
                ligand_embeddings = np.append(ligand_embeddings, ligand_embed.cpu().detach().numpy(), axis = None)
                receptor_embeddings = np.append(receptor_embeddings, receptor_embed.cpu().detach().numpy(), axis = None)
                joint_embeddings = np.append(joint_embeddings, joint_embed.cpu().detach().numpy(), axis = None)

    acc = correct_samples / total_samples
    loss /= total_samples
    
    if store_embeddings:
        writeEmbeddingsIntoFile(f'{log_dir}/{mode}_logs/{model_name}/embeddings/', f'ligand_embeddings.csv', all_ids, ligand_embeddings)
        writeEmbeddingsIntoFile(f'{log_dir}/{mode}_logs/{model_name}/embeddings/', f'receptor_embeddings.csv', all_ids, receptor_embeddings)
        writeEmbeddingsIntoFile(f'{log_dir}/{mode}_logs/{model_name}/embeddings/', f'joint_embeddings.csv', all_ids, joint_embeddings)
    if store_predictions:
        writePredictionsIntoFile(f'{log_dir}/{mode}_logs/{model_name}/predictions/', f'predictions.csv', all_ids, all_y_pred, all_y_actual, all_y_p1)
        if mode == 'test':
            plotCalibration(log_dir, model_name)

    end_time = time.time()
    t = end_time - start_time
    print(mode, 'Loss: %2.5e Accuracy: %2.5f Epoch Time: %2.5f' %(loss,acc,t))
    if mode == 'test':
        print(f'Balanced Accuracy: {round(balanced_accuracy_score(all_y_actual, all_y_pred) * 100, 4)} %')
        print(classification_report(all_y_actual, all_y_pred))
        print(confusion_matrix(all_y_actual, all_y_pred))
        
    if mode == 'validation':
        Net.train()
    return loss, acc


def plot_loss_acc(losses, accs, val_losses, val_accs, log_dir, model_name):
    file_path = f'{log_dir}/train_logs/{model_name}/figures'
    makeDir(file_path)
    plt.plot(np.array(accs),color='red', label='Train accuracy')
    plt.plot(np.array(val_accs),color='blue', label='Val accuracy')
    plt.legend()
    plt.savefig(f'{file_path}/acc.pdf', format="pdf", bbox_inches="tight")
    plt.clf()
    plt.plot(np.array(losses),color='red', label='Train loss')
    plt.plot(np.array(val_losses),color='blue', label='Val loss')
    plt.legend()
    plt.savefig(f'{file_path}/loss.pdf', format="pdf", bbox_inches="tight")
    plt.clf()
    plt.show()
    return

def plotCalibration(log_dir, model_name):
    df = pd.read_csv(f'{log_dir}/test_logs/{model_name}/predictions/predictions.csv')
    y_probs = df['P1'].values
    y_test = df['Actual Label'].values
    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', linestyle='-', color='b')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.savefig(f'{log_dir}/test_logs/{model_name}/calibration.pdf', format="pdf", bbox_inches="tight") 
    plt.clf()

def makeDir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
def writeEmbeddingsIntoFile(file_path, file_name, ids, embeddings):
    if not os.path.exists(file_path):
        os.makedirs(file_path) 
    with open(file_path + file_name, mode='w', newline='') as file:
        writer = csv.writer(file) 
        for row in range(ids.shape[0]):
            writer.writerow(np.concatenate(([ids[row]], embeddings[row])))

def writePredictionsIntoFile(file_path, file_name, ids, y_pred, y_actual, y_p1):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(file_path + file_name, mode='w', newline='') as file:
        writer = csv.writer(file) 
        writer.writerow(['ID', 'Predicted Label', 'Actual Label', 'P1'])
        for row in range(ids.shape[0]):
            writer.writerow([ids[row], y_pred[row], y_actual[row], y_p1[row]])

def pickleDump(file_path, file_name, content):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(file_path + file_name, 'wb') as f:
        pickle.dump(content, f)

def pickleRead(file_path, file_name):
    if (not os.path.exists(file_path)) or ( not os.path.exists(file_path + file_name)):
        return None
    with open(file_path + file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def writeIntoFile(file_path, file_name, content, onlyIds = False):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(file_path + file_name, mode='w', newline='') as file:
        writer = csv.writer(file) 
        if onlyIds == True:
            writer.writerow(['IDs'])
            for row in range(len(content)):
                writer.writerow([content[row]])
        else:
            for row in range(len(content)):
                writer.writerow(content[row])

def saveModel(model, file_path, model_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    torch.save(model.state_dict(), f"{file_path}/{model_name}.pt")