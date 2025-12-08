from myImports import *

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


def convert_to_list(lst):
    ans = []
    for i in range(len(lst)):
        ans.append(lst[i].item())
    return ans

def trainModel(model, train_loader, val_loader, fold_num, 
               weights_file_path, train_text_file_path, val_text_file_path, test_text_file_path, 
               train_key_embedding_file_path, train_lock_embedding_file_path, train_concat_embedding_file_path, 
               val_key_embedding_file_path, val_lock_embedding_file_path, val_concat_embedding_file_path):
    
    # These are the parameters
    
    NUM_EPOCHS = 100
    LOSS_CRITERION = nn.CrossEntropyLoss()
    LEARNING_RATE = 1e-5
    OPTIMIZER = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, patience = 2, verbose = True)


    loss_train = []
    loss_val = []

    acc_train = []
    acc_val = []

    train_list = []
    val_list = []


    best_val_acc = -1
    best_epoch = 1

    for epoch in range(NUM_EPOCHS):
        model.train()
        print(f'Epoch: {epoch + 1}')

        batch_loss_train = 0 
        average_batch_loss_train = 0

        train_correct = 0
        train_samples = 0
    
        train_predicted_label = []
        train_actual_label = []
        train_pred_proba = []

        train_key_embeddings = []
        train_lock_embeddings = []
        train_concat_embeddings = []

        for i, data in enumerate(train_loader, 0):
            OPTIMIZER.zero_grad()

            ids, inputs, labels = data
            k1, k2, k3, k4, k5, l1, l2, l3, l4 = inputs
            k1, k2, k3, k4, k5, l1, l2, l3, l4 = k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), l1.to(device), l2.to(device), l3.to(device), l4.to(device)
            labels = labels.to(device)
            

            outputs, key_embed, lock_embed, concat_embed = model.forward(k1, k2, k3, k4, k5, l1, l2, l3, l4)
            loss = LOSS_CRITERION(outputs, labels)

            loss.backward()
            OPTIMIZER.step()
            
            outputs = nn.Softmax(dim = 1)(outputs)
            _, prediction = torch.max(outputs, dim = 1)

            train_correct += (prediction == labels).sum().item()
            train_samples += len(outputs)

            train_predicted_label.extend(prediction)
            train_actual_label.extend(labels)
            train_pred_proba.extend(outputs[:, 1])
            # train_key_embeddings.extend(key_embed)
            # train_lock_embeddings.extend(lock_embed)
            # train_concat_embeddings.extend(concat_embed)

            _loss = loss.item()
            batch_loss_train += _loss
            average_batch_loss_train = batch_loss_train / (i+1)
            # print(f'In epoch {epoch} Current batch loss: {_loss}, average train batch loss: {average_batch_loss_train}')
        
        loss_train.append(average_batch_loss_train)
        SCHEDULER.step(average_batch_loss_train)

        train_acc = (train_correct / train_samples) * 100
        acc_train.append(train_acc)

        print(f'Training loss: {round(average_batch_loss_train, 3)} Training acc: {round(train_acc, 3)} ')
        torch.save(model.state_dict(), f"{weights_file_path}/epoch_{epoch+1}.pt")
        
        train_actual_lbl = []
        train_predicted_lbl = []
        train_pred_prob = []
        key_emb = []
        lock_emb = []
        concat_emb = []
        
        for i in range(len(train_actual_label)):
            train_actual_lbl.append(train_actual_label[i].item())
            train_predicted_lbl.append(train_predicted_label[i].item())
            train_pred_prob.append(train_pred_proba[i].item())
            # key_emb.append(convert_to_list(train_key_embeddings[i]))
            # lock_emb.append(convert_to_list(train_lock_embeddings[i]))
            # concat_emb.append(convert_to_list(train_concat_embeddings[i]))
        
        temp = []

        for i in range(len(train_actual_lbl)):
            lst = []
            lst.append(train_actual_lbl[i])
            lst.append(train_predicted_lbl[i])
            lst.append(train_pred_prob[i])
            temp.append(lst)


        # with open(train_key_embedding_file_path + f'epoch_{epoch}.csv', mode='w', newline='') as file:
        #     writer = csv.writer(file) 
        #     for row in range(len(key_emb)):
        #         writer.writerow(key_emb[row])

        # with open(train_lock_embedding_file_path + f'epoch_{epoch}.csv', mode='w', newline='') as file:
        #     writer = csv.writer(file) 
        #     for row in range(len(lock_emb)):
        #         writer.writerow(lock_emb[row])
        
        # with open(train_concat_embedding_file_path + f'epoch_{epoch}.csv', mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     for row in range(len(concat_emb)):
        #         writer.writerow(concat_emb[row])
                

        with open(train_text_file_path + f'epoch_{epoch}.csv', mode='w', newline='') as file:
            writer = csv.writer(file) 
            writer.writerow(['Actual Label', 'Predicted Label', 'P1'])
            for row in range(len(temp)):
                writer.writerow(temp[row])

        


        ## Validation
                
        if val_loader is not None:
            val_predicted_label = []
            val_actual_label = []
            val_pred_proba = []

            val_key_embeddings = []
            val_lock_embeddings = []
            val_concat_embeddings = []


            batch_loss_val = 0
            avg_loss_val = 0

            val_correct = 0
            val_samples = 0
            avg_loss_val = 0

            with torch.no_grad():
                model.eval()
                for i, data in enumerate(val_loader):
                    ids, inputs, labels = data
                    k1, k2, k3, k4, k5, l1, l2, l3, l4 = inputs
                    k1, k2, k3, k4, k5, l1, l2, l3, l4 = k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), l1.to(device), l2.to(device), l3.to(device), l4.to(device)
                    labels = labels.to(device)

                    outputs, key_embed, lock_embed, concat_embed = model.forward(k1, k2, k3, k4, k5, l1, l2, l3, l4)
                    loss = LOSS_CRITERION(outputs, labels)

                    _loss_val = loss.item()
                    batch_loss_val += _loss_val
                    avg_loss_val = batch_loss_val / (i+1)

                    outputs = nn.Softmax(dim = 1)(outputs)
                    _, prediction = torch.max(outputs, dim = 1)

                    val_correct += (prediction == labels).sum().item()
                    val_samples += len(outputs)

                    val_predicted_label.extend(prediction)
                    val_actual_label.extend(labels)
                    val_pred_proba.extend(outputs[:, 1])
                    # val_key_embeddings.extend(key_embed)
                    # val_lock_embeddings.extend(lock_embed)
                    # val_concat_embeddings.extend(concat_embed)

                loss_val.append(avg_loss_val)
                
                val_acc = (val_correct / val_samples) * 100
                acc_val.append(val_acc)
                
                print(f'val loss: {round(avg_loss_val, 3)}, val acc: {round(val_acc, 3)}')

                val_actual_lbl = []
                val_predicted_lbl = []
                val_pred_prob = []
                key_emb = []
                lock_emb = []
                concat_emb = []
                
                for i in range(len(val_actual_label)):
                    val_actual_lbl.append(val_actual_label[i].item())
                    val_predicted_lbl.append(val_predicted_label[i].item())
                    val_pred_prob.append(val_pred_proba[i].item())
                    # key_emb.append(convert_to_list(val_key_embeddings[i][0]))
                    # lock_emb.append(convert_to_list(val_lock_embeddings[i][0]))
                    # concat_emb.append(convert_to_list(val_concat_embeddings[i]))

                temp = []

                for i in range(len(val_actual_lbl)):
                    lst = []
                    lst.append(val_actual_lbl[i])
                    lst.append(val_predicted_lbl[i])
                    lst.append(val_pred_prob[i])
                    temp.append(lst)

                # with open(val_key_embedding_file_path + f'epoch_{epoch}.csv', mode='w', newline='') as file:
                #     writer = csv.writer(file) 
                #     for row in range(len(key_emb)):
                #         writer.writerow(key_emb[row])

                # with open(val_lock_embedding_file_path + f'epoch_{epoch}.csv', mode='w', newline='') as file:
                #     writer = csv.writer(file) 
                #     for row in range(len(lock_emb)):
                #         writer.writerow(lock_emb[row])
                
                # with open(val_concat_embedding_file_path + f'epoch_{epoch}.csv', mode='w', newline='') as file:
                #     writer = csv.writer(file)
                #     for row in range(len(concat_emb)):
                #         writer.writerow(concat_emb[row])

                with open(val_text_file_path + f'epoch_{epoch}.csv', mode='w', newline='') as file:
                    writer = csv.writer(file) 
                    writer.writerow(['Actual Label', 'Predicted Label', 'P1'])
                    for row in range(len(temp)):
                        writer.writerow(temp[row])

                if round(val_acc, 3) >= best_val_acc:
                    best_val_acc = round(val_acc, 3)
                    best_epoch = epoch+1

    print('Training FINISHED')

    return model, train_list, val_list, best_epoch, best_val_acc, loss_train, loss_val, acc_train, acc_val


def testModel(model, test_loader, test_text_file_path , test_key_embedding_file_path, test_lock_embedding_file_path, test_concat_embedding_file_path, testingOn = ''):
    test_correct = 0
    test_samples = 0

    predicted_label = []
    actual_label = []
    predicted_proba = []

    test_key_embeddings = []
    test_lock_embeddings = []
    test_concat_embeddings = []

    test_ids_0 = []
    test_ids_1 = []
    test_ids_2 = []

    with torch.no_grad():
        batch_num = 0
        for data in test_loader:
            LOSS_CRITERION = nn.CrossEntropyLoss()

            ids, inputs, labels = data
            # print(f'batch: {batch_num} {ids}')
            # batch_num += 1
            k1, k2, k3, k4, k5, l1, l2, l3, l4 = inputs
            k1, k2, k3, k4, k5, l1, l2, l3, l4 = k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), l1.to(device), l2.to(device), l3.to(device), l4.to(device)
            # labels = labels.to(device)
            
            outputs , key_embed, lock_embed, concat_embed  = model.forward(k1, k2, k3, k4, k5, l1, l2, l3, l4)
            # loss = LOSS_CRITERION(outputs, labels)

            outputs = nn.Softmax(dim = 1)(outputs)
            _, prediction = torch.max(outputs, dim = 1)
            
            # test_correct += (prediction == labels).sum().item()
            test_samples += len(outputs)

            predicted_label.extend(prediction)
            # actual_label.extend(labels)
            predicted_proba.extend(outputs[:, 1])
            test_key_embeddings.extend(key_embed)
            test_lock_embeddings.extend(lock_embed)
            test_concat_embeddings.extend(concat_embed)
            test_ids_0.extend(ids)
            
    # test_acc = (test_correct / test_samples) * 100
    # print(f'Accuracy on {testingOn} set: {round(test_acc, 4)} %')


    actual_lbl = []
    predicted_lbl = []
    pred_prob = []
    key_emb = []
    lock_emb = []
    concat_emb = []

    for i in range(len(predicted_label)):
        # actual_lbl.append(actual_label[i].item())
        predicted_lbl.append(predicted_label[i].item())
        pred_prob.append(predicted_proba[i].item())
        key_emb.append(convert_to_list(test_key_embeddings[i]))
        lock_emb.append(convert_to_list(test_lock_embeddings[i]))
        concat_emb.append(convert_to_list(test_concat_embeddings[i]))


    # f1 = f1_score(actual_lbl, predicted_lbl, average = 'macro')
    # print(f'F1 score on {testingOn} set: {f1}')
    # print(f'Balanced Accuracy on {testingOn} set: {round(balanced_accuracy_score(actual_lbl, predicted_lbl) * 100, 4)} %')
    # print(classification_report(actual_lbl, predicted_lbl))
    # print(confusion_matrix(actual_lbl, predicted_lbl))

    temp = []

    for i in range(len(test_ids_0)):
        lst = []
        lst.append(test_ids_0[i])
        # lst.append(actual_lbl[i])
        lst.append(predicted_lbl[i])
        lst.append(pred_prob[i])
        
        temp.append(lst)

    if not os.path.exists(test_key_embedding_file_path):
        print('making key file')
        os.makedirs(test_key_embedding_file_path)
    if not os.path.exists(test_lock_embedding_file_path):
        print('making lock file')
        os.makedirs(test_lock_embedding_file_path)
    if not os.path.exists(test_concat_embedding_file_path):
        print('making concat file')
        os.makedirs(test_concat_embedding_file_path)
    if not os.path.exists(test_text_file_path):
        print('making text file')
        os.makedirs(test_text_file_path)


    with open(test_key_embedding_file_path + f'Ligand_Embeddings.csv', mode='w', newline='') as file:
        writer = csv.writer(file) 
        for row in range(len(key_emb)):
            writer.writerow(np.concatenate(([test_ids_0[row]], key_emb[row])))

    with open(test_lock_embedding_file_path + f'Receptor_Embeddings.csv', mode='w', newline='') as file:
        writer = csv.writer(file) 
        for row in range(len(lock_emb)):
            writer.writerow(np.concatenate(([test_ids_0[row]], lock_emb[row])))
    
    with open(test_concat_embedding_file_path + f'LR_Pair_Embeddings.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in range(len(concat_emb)):
            writer.writerow(np.concatenate(([test_ids_0[row]], concat_emb[row])))

    with open(test_text_file_path + f'{testingOn}.csv', mode='w', newline='') as file:
        writer = csv.writer(file) 
        writer.writerow(['ID', 'Predicted Label', 'P1'])
        for row in range(len(temp)):
            writer.writerow(temp[row])

    return predicted_lbl, pred_prob


def plotROC(train_actual_lbl, train_pred_prob, test_actual_lbl, test_pred_prob, choice):
    train_fpr, train_tpr, _ = roc_curve(train_actual_lbl, train_pred_prob)
    train_roc_auc = auc(train_fpr, train_tpr)

    test_fpr, test_tpr, _ = roc_curve(test_actual_lbl, test_pred_prob)
    test_roc_auc = auc(test_fpr, test_tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(train_fpr, train_tpr, color='darkorange', lw=2, label=f'Training ROC curve (AUC = {train_roc_auc:.2f})')
    plt.plot(test_fpr, test_tpr, color='blue', lw=2, label=f'Test ROC curve (AUC = {test_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'../figures/Evolf ROC {choice}.pdf', format="pdf", bbox_inches="tight") 
    plt.show()