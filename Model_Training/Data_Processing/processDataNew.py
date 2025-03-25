from myImports import *
from myDataset import *
from utils import *


def getIds(dataType, fold_num, makeFolds = False):
    # will overwrite the previous splitting ids
    if dataType == 'evolf':
        main = pd.read_csv('../all_data/raw_data/TrainDataRaw/25_EvOlf_MainDataset.csv')
    elif dataType == 'glass':
        main = pd.read_csv('../all_data/raw_data/TrainDataRaw/25_GLASS_MainDataset.csv')

    Human = main[main["Species"] == "Human"]["IDs"].values.tolist()
    Others = main[main["Species"] != "Human"]["IDs"].values.tolist()

    random.shuffle(Human)
    random.shuffle(Others)

    len_human = len(Human)

    human_train_ids = Human[:int(0.8 * len_human)]
    human_val_ids = Human[int(0.8 * len_human): int(0.9 * len_human)]
    human_test_ids = Human[int(0.9 * len_human) : ]

    len_others = len(Others)

    others_train = Others[:int(0.8 * len_others)]
    others_val = Others[int(0.8 * len_others): int(0.9 * len_others)]
    others_test = Others[int(0.9 * len_others) : ]

    all_train_ids = np.concatenate((others_train, human_train_ids), axis=0)
    all_val_ids = np.concatenate((others_val, human_val_ids), axis=0)
    all_test_ids = np.concatenate((others_test, human_test_ids), axis=0)

    
    if makeFolds:
        save_path = f'../all_data/processedData/MainModel/10Fold/fold_{fold_num}/ids/'
    else:
        save_path = f'../all_data/processedData/MainModel/SingleDataset/ids/'

    writeIntoFile(save_path, f'{dataType}_all_train_ids.csv', all_train_ids, onlyIds=True)
    pickleDump(save_path, f'{dataType}_all_train_ids.pkl', all_train_ids)

    writeIntoFile(save_path, f'{dataType}_all_val_ids.csv', all_val_ids, onlyIds=True)
    pickleDump(save_path, f'{dataType}_all_val_ids.pkl', all_val_ids)

    writeIntoFile(save_path, f'{dataType}_all_test_ids.csv', all_test_ids, onlyIds=True)
    pickleDump(save_path, f'{dataType}_all_test_ids.pkl', all_test_ids)

    writeIntoFile(save_path, f'{dataType}_human_train_ids.csv', human_train_ids, onlyIds=True)
    pickleDump(save_path, f'{dataType}_human_train_ids.pkl', human_train_ids)

    writeIntoFile(save_path, f'{dataType}_human_val_ids.csv', human_val_ids, onlyIds=True)
    pickleDump(save_path, f'{dataType}_human_val_ids.pkl', human_val_ids)

    writeIntoFile(save_path, f'{dataType}_human_test_ids.csv', human_test_ids, onlyIds=True)
    pickleDump(save_path, f'{dataType}_human_test_ids.pkl', human_test_ids)

    return all_train_ids, all_val_ids, all_test_ids, human_train_ids, human_val_ids, human_test_ids

def getCombIds(fold_num, makeFolds):

    dataType ='comb'

    if makeFolds:
        save_path = f'../all_data/processedData/MainModel/10Fold/fold_{fold_num}/ids/'
    else:
        save_path = f'../all_data/processedData/MainModel/SingleDataset/ids/'


    evolf_all_train_ids = pickleRead(save_path, f'evolf_all_train_ids.pkl')
    evolf_all_val_ids = pickleRead(save_path, f'evolf_all_val_ids.pkl')
    evolf_all_test_ids = pickleRead(save_path, f'evolf_all_test_ids.pkl')

    glass_all_train_ids = pickleRead(save_path, f'glass_all_train_ids.pkl')
    glass_all_val_ids = pickleRead(save_path, f'glass_all_val_ids.pkl')
    glass_all_test_ids = pickleRead(save_path, f'glass_all_test_ids.pkl')

    evolf_human_train_ids = pickleRead(save_path, f'evolf_human_train_ids.pkl')
    evolf_human_val_ids = pickleRead(save_path, f'evolf_human_val_ids.pkl')
    evolf_human_test_ids = pickleRead(save_path, f'evolf_human_test_ids.pkl')

    glass_human_train_ids = pickleRead(save_path, f'glass_human_train_ids.pkl')
    glass_human_val_ids = pickleRead(save_path, f'glass_human_val_ids.pkl')
    glass_human_test_ids = pickleRead(save_path, f'glass_human_test_ids.pkl')

    all_train_ids = np.concatenate((evolf_all_train_ids, glass_all_train_ids), axis=None)
    all_val_ids = np.concatenate((evolf_all_val_ids, glass_all_val_ids), axis=None)
    all_test_ids = np.concatenate((evolf_all_test_ids, glass_all_test_ids), axis=None)

    human_train_ids = np.concatenate((evolf_human_train_ids, glass_human_train_ids), axis=None)
    human_val_ids = np.concatenate((evolf_human_val_ids, glass_human_val_ids), axis=None)
    human_test_ids = np.concatenate((evolf_human_test_ids, glass_human_test_ids), axis=None)

    
    if makeFolds:
        save_path = f'../all_data/processedData/MainModel/10Fold/fold_{fold_num}/ids/'
    else:
        save_path = f'../all_data/processedData/MainModel/SingleDataset/ids/'

    writeIntoFile(save_path, f'{dataType}_all_train_ids.csv', all_train_ids, onlyIds=True)
    pickleDump(save_path, f'{dataType}_all_train_ids.pkl', all_train_ids)

    writeIntoFile(save_path, f'{dataType}_all_val_ids.csv', all_val_ids, onlyIds=True)
    pickleDump(save_path, f'{dataType}_all_val_ids.pkl', all_val_ids)

    writeIntoFile(save_path, f'{dataType}_all_test_ids.csv', all_test_ids, onlyIds=True)
    pickleDump(save_path, f'{dataType}_all_test_ids.pkl', all_test_ids)

    writeIntoFile(save_path, f'{dataType}_human_train_ids.csv', human_train_ids, onlyIds=True)
    pickleDump(save_path, f'{dataType}_human_train_ids.pkl', human_train_ids)

    writeIntoFile(save_path, f'{dataType}_human_val_ids.csv', human_val_ids, onlyIds=True)
    pickleDump(save_path, f'{dataType}_human_val_ids.pkl', human_val_ids)

    writeIntoFile(save_path, f'{dataType}_human_test_ids.csv', human_test_ids, onlyIds=True)
    pickleDump(save_path, f'{dataType}_human_test_ids.pkl', human_test_ids)

    return all_train_ids, all_val_ids, all_test_ids, human_train_ids, human_val_ids, human_test_ids



def processTrainData(dataType, dataSubtype, fold_num, makeFolds):

    mypath = '../all_data/raw_data/TrainDataRaw/'

    if dataType == 'evolf':
        file_name = 'EvOlf'
    elif dataType == 'glass':
        file_name = 'GLASS'
    else:
        file_name = 'COMB'
    g2v = pd.read_csv(f'{mypath}25_{file_name}_Graph2Vec_Final.csv')
    sig = pd.read_csv(f'{mypath}25_{file_name}_Signaturizer_Final.csv')
    mord = pd.read_csv(f'{mypath}25_{file_name}_Mordred_Final.csv')
    m2v = pd.read_csv(f'{mypath}25_{file_name}_Mol2Vec_Final.csv')
    cb = pd.read_csv(f'{mypath}25_{file_name}_ChemBERTa_Final.csv')
    main = pd.read_csv(f'{mypath}25_{file_name}_MainDataset.csv')
    bfd = pd.read_csv(f'{mypath}25_{file_name}_ProtBFD_Final.csv')
    t5 = pd.read_csv(f'{mypath}25_{file_name}_ProtT5_Final.csv')
    mf = pd.read_csv(f'{mypath}25_{file_name}_MathFeature_Final.csv')
    r = pd.read_csv(f'{mypath}25_{file_name}_ProtR_Final.csv')

    
    sort_order = pd.Series(main.index, index=main['IDs']).to_dict()

    Ligand_1 = pd.merge(main, g2v, on='Ligand_ID', how='inner')
    Ligand_2 = pd.merge(main, sig, on='Ligand_ID', how='inner')
    Ligand_3 = pd.merge(main, mord, on='Ligand_ID', how='inner')
    Ligand_4 = pd.merge(main, m2v, on='Ligand_ID', how='inner')
    Ligand_5 = pd.merge(main, cb, on='Ligand_ID', how='inner')

    Receptor_1 = pd.merge(main, bfd, on='Receptor_ID', how='inner')
    Receptor_2 = pd.merge(main, t5, on='Receptor_ID', how='inner')
    Receptor_3 = pd.merge(main, mf, on='Receptor_ID', how='inner')
    Receptor_4 = pd.merge(main, r, on='Receptor_ID', how='inner')

    Ligand_1['sort_order'] = Ligand_1['IDs'].map(sort_order)
    Ligand_2['sort_order'] = Ligand_2['IDs'].map(sort_order)
    Ligand_3['sort_order'] = Ligand_3['IDs'].map(sort_order)
    Ligand_4['sort_order'] = Ligand_4['IDs'].map(sort_order)
    Ligand_5['sort_order'] = Ligand_5['IDs'].map(sort_order)

    Receptor_1['sort_order'] = Receptor_1['IDs'].map(sort_order)
    Receptor_2['sort_order'] = Receptor_2['IDs'].map(sort_order)
    Receptor_3['sort_order'] = Receptor_3['IDs'].map(sort_order)
    Receptor_4['sort_order'] = Receptor_4['IDs'].map(sort_order)

    Ligand_1.sort_values(by='sort_order', inplace=True)
    Ligand_1.drop(columns=['sort_order'], inplace=True)
    Ligand_1.reset_index(drop=True, inplace=True)

    Ligand_2.sort_values(by='sort_order', inplace=True)
    Ligand_2.drop(columns=['sort_order'], inplace=True)
    Ligand_2.reset_index(drop=True, inplace=True)

    Ligand_3.sort_values(by='sort_order', inplace=True)
    Ligand_3.drop(columns=['sort_order'], inplace=True)
    Ligand_3.reset_index(drop=True, inplace=True)

    Ligand_4.sort_values(by='sort_order', inplace=True)
    Ligand_4.drop(columns=['sort_order'], inplace=True)
    Ligand_4.reset_index(drop=True, inplace=True)

    Ligand_5.sort_values(by='sort_order', inplace=True)
    Ligand_5.drop(columns=['sort_order'], inplace=True)
    Ligand_5.reset_index(drop=True, inplace=True)


    Receptor_1.sort_values(by='sort_order', inplace=True)
    Receptor_1.drop(columns=['sort_order'], inplace=True)
    Receptor_1.reset_index(drop=True, inplace=True)

    Receptor_2.sort_values(by='sort_order', inplace=True)
    Receptor_2.drop(columns=['sort_order'], inplace=True)
    Receptor_2.reset_index(drop=True, inplace=True)

    Receptor_3.sort_values(by='sort_order', inplace=True)
    Receptor_3.drop(columns=['sort_order'], inplace=True)
    Receptor_3.reset_index(drop=True, inplace=True)

    Receptor_4.sort_values(by='sort_order', inplace=True)
    Receptor_4.drop(columns=['sort_order'], inplace=True)
    Receptor_4.reset_index(drop=True, inplace=True)
    
    if makeFolds:
        save_path = f'../all_data/processedData/MainModel/10Fold/fold_{fold_num}/ids/'
    else:
        save_path = f'../all_data/processedData/MainModel/SingleDataset/ids/'

    train_ids = pd.read_csv(save_path + f'{dataType}_{dataSubtype}_train_ids.csv')
    sort_order = pd.Series(main.index, index=main['IDs']).to_dict()
    main = pd.merge(main, train_ids, on=['IDs'], how='inner')

    Ligand_1 = pd.merge(main, Ligand_1, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Ligand_2 = pd.merge(main, Ligand_2, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Ligand_3 = pd.merge(main, Ligand_3, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Ligand_4 = pd.merge(main, Ligand_4, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Ligand_5 = pd.merge(main, Ligand_5, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')

    Receptor_1 = pd.merge(main, Receptor_1, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Receptor_2 = pd.merge(main, Receptor_2, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Receptor_3 = pd.merge(main, Receptor_3, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Receptor_4 = pd.merge(main, Receptor_4, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')

    Ligand_1['sort_order'] = Ligand_1['IDs'].map(sort_order)
    Ligand_2['sort_order'] = Ligand_2['IDs'].map(sort_order)
    Ligand_3['sort_order'] = Ligand_3['IDs'].map(sort_order)
    Ligand_4['sort_order'] = Ligand_4['IDs'].map(sort_order)
    Ligand_5['sort_order'] = Ligand_5['IDs'].map(sort_order)

    Receptor_1['sort_order'] = Receptor_1['IDs'].map(sort_order)
    Receptor_2['sort_order'] = Receptor_2['IDs'].map(sort_order)
    Receptor_3['sort_order'] = Receptor_3['IDs'].map(sort_order)
    Receptor_4['sort_order'] = Receptor_4['IDs'].map(sort_order)

    Ligand_1.sort_values(by='sort_order', inplace=True)
    Ligand_1.drop(columns=['sort_order'], inplace=True)
    Ligand_1.reset_index(drop=True, inplace=True)

    Ligand_2.sort_values(by='sort_order', inplace=True)
    Ligand_2.drop(columns=['sort_order'], inplace=True)
    Ligand_2.reset_index(drop=True, inplace=True)

    Ligand_3.sort_values(by='sort_order', inplace=True)
    Ligand_3.drop(columns=['sort_order'], inplace=True)
    Ligand_3.reset_index(drop=True, inplace=True)

    Ligand_4.sort_values(by='sort_order', inplace=True)
    Ligand_4.drop(columns=['sort_order'], inplace=True)
    Ligand_4.reset_index(drop=True, inplace=True)

    Ligand_5.sort_values(by='sort_order', inplace=True)
    Ligand_5.drop(columns=['sort_order'], inplace=True)
    Ligand_5.reset_index(drop=True, inplace=True)


    Receptor_1.sort_values(by='sort_order', inplace=True)
    Receptor_1.drop(columns=['sort_order'], inplace=True)
    Receptor_1.reset_index(drop=True, inplace=True)

    Receptor_2.sort_values(by='sort_order', inplace=True)
    Receptor_2.drop(columns=['sort_order'], inplace=True)
    Receptor_2.reset_index(drop=True, inplace=True)

    Receptor_3.sort_values(by='sort_order', inplace=True)
    Receptor_3.drop(columns=['sort_order'], inplace=True)
    Receptor_3.reset_index(drop=True, inplace=True)

    Receptor_4.sort_values(by='sort_order', inplace=True)
    Receptor_4.drop(columns=['sort_order'], inplace=True)
    Receptor_4.reset_index(drop=True, inplace=True)


    Ligand_1 =  np.array(Ligand_1.iloc[:, 6:])
    Ligand_2 =  np.array(Ligand_2.iloc[:, 6:])
    Ligand_3 =  np.array(Ligand_3.iloc[:, 6:])
    Ligand_4 =  np.array(Ligand_4.iloc[:, 6:])
    Ligand_5 =  np.array(Ligand_5.iloc[:, 6:])

    Receptor_1 = np.array(Receptor_1.iloc[:, 5:])
    Receptor_2 = np.array(Receptor_2.iloc[:, 5:])
    Receptor_3 = np.array(Receptor_3.iloc[:, 5:])
    Receptor_4 = np.array(Receptor_4.iloc[:, 5:])

    
    if makeFolds:
        scaler_path = f'../all_data/processingModels/10Folds/fold_{fold_num}/scaler_models'
    else:
        scaler_path = '../all_data/processingModels/scaler_models'

    scaler_Ligand_1 = StandardScaler()
    Ligand_1 = scaler_Ligand_1.fit_transform(Ligand_1)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Ligand_1.pkl', scaler_Ligand_1)

    scaler_Ligand_2 = StandardScaler()
    Ligand_2 = scaler_Ligand_2.fit_transform(Ligand_2)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Ligand_2.pkl', scaler_Ligand_2)

    scaler_Ligand_3 = StandardScaler()
    Ligand_3 = scaler_Ligand_3.fit_transform(Ligand_3)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Ligand_3.pkl', scaler_Ligand_3)

    scaler_Ligand_4 = StandardScaler()
    Ligand_4 = scaler_Ligand_4.fit_transform(Ligand_4)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Ligand_4.pkl', scaler_Ligand_4)

    scaler_Ligand_5 = StandardScaler()
    Ligand_5 = scaler_Ligand_5.fit_transform(Ligand_5)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Ligand_5.pkl', scaler_Ligand_5)


    scaler_Receptor_1 = StandardScaler()
    Receptor_1 = scaler_Receptor_1.fit_transform(Receptor_1)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Receptor_1.pkl', scaler_Receptor_1)

    scaler_Receptor_2 = StandardScaler()
    Receptor_2 = scaler_Receptor_2.fit_transform(Receptor_2)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Receptor_2.pkl', scaler_Receptor_2)

    scaler_Receptor_3 = StandardScaler()
    Receptor_3 = scaler_Receptor_3.fit_transform(Receptor_3)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Receptor_3.pkl', scaler_Receptor_3)

    scaler_Receptor_4 = StandardScaler()
    Receptor_4 = scaler_Receptor_4.fit_transform(Receptor_4)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Receptor_4.pkl', scaler_Receptor_4)

    if makeFolds:
        pca_path = f'../all_data/processingModels/10Folds/fold_{fold_num}/pca_models'
    else:
        pca_path = '../all_data/processingModels/pca_models'

    pca_Ligand_1 = IncrementalPCA(n_components=128)
    Ligand_1 = pca_Ligand_1.fit_transform(Ligand_1)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Ligand_1.pkl', pca_Ligand_1)

    pca_Ligand_2 = IncrementalPCA(n_components=128)
    Ligand_2 = pca_Ligand_2.fit_transform(Ligand_2)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Ligand_2.pkl', pca_Ligand_2)


    pca_Ligand_3 = IncrementalPCA(n_components=128)
    Ligand_3 = pca_Ligand_3.fit_transform(Ligand_3)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Ligand_3.pkl', pca_Ligand_3)


    pca_Ligand_4 = IncrementalPCA(n_components=128)
    Ligand_4 = pca_Ligand_4.fit_transform(Ligand_4)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Ligand_4.pkl', pca_Ligand_4)


    pca_Ligand_5 = IncrementalPCA(n_components=128)
    Ligand_5 = pca_Ligand_5.fit_transform(Ligand_5)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Ligand_5.pkl', pca_Ligand_5)


    pca_Receptor_1 = IncrementalPCA(n_components=128)
    Receptor_1 = pca_Receptor_1.fit_transform(Receptor_1)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Receptor_1.pkl', pca_Receptor_1)


    pca_Receptor_2 = IncrementalPCA(n_components=128)
    Receptor_2 = pca_Receptor_2.fit_transform(Receptor_2)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Receptor_2.pkl', pca_Receptor_2)

    pca_Receptor_3 = IncrementalPCA(n_components=128)
    Receptor_3 = pca_Receptor_3.fit_transform(Receptor_3)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Receptor_3.pkl', pca_Receptor_3)

    pca_Receptor_4 = IncrementalPCA(n_components=128)
    Receptor_4 = pca_Receptor_4.fit_transform(Receptor_4)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Receptor_4.pkl', pca_Receptor_4)

    
    Ligand_1 = np.array(Ligand_1, dtype = np.float32)
    Ligand_2 = np.array(Ligand_2, dtype = np.float32)
    Ligand_3 = np.array(Ligand_3, dtype = np.float32)
    Ligand_4 = np.array(Ligand_4, dtype = np.float32)
    Ligand_5 = np.array(Ligand_5, dtype = np.float32)

    Receptor_1 = np.array(Receptor_1, dtype = np.float32)
    Receptor_2 = np.array(Receptor_2, dtype = np.float32)
    Receptor_3 = np.array(Receptor_3, dtype = np.float32)
    Receptor_4 = np.array(Receptor_4, dtype = np.float32)

    Ligand_1 = Ligand_1.reshape(Ligand_1.shape[0], 1, Ligand_1.shape[1])
    Ligand_2 = Ligand_2.reshape(Ligand_2.shape[0], 1, Ligand_2.shape[1])
    Ligand_3 = Ligand_3.reshape(Ligand_3.shape[0], 1, Ligand_3.shape[1])
    Ligand_4 = Ligand_4.reshape(Ligand_4.shape[0], 1, Ligand_4.shape[1])
    Ligand_5 = Ligand_5.reshape(Ligand_5.shape[0], 1, Ligand_5.shape[1])

    Receptor_1 = Receptor_1.reshape(Receptor_1.shape[0], 1, Receptor_1.shape[1])
    Receptor_2 = Receptor_2.reshape(Receptor_2.shape[0], 1, Receptor_2.shape[1])
    Receptor_3 = Receptor_3.reshape(Receptor_3.shape[0], 1, Receptor_3.shape[1])
    Receptor_4 = Receptor_4.reshape(Receptor_4.shape[0], 1, Receptor_4.shape[1])


    Ligand_dict = {}

    Ligand_dict[0] = Ligand_1
    Ligand_dict[1] = Ligand_2
    Ligand_dict[2] = Ligand_3
    Ligand_dict[3] = Ligand_4
    Ligand_dict[4] = Ligand_5

    Receptor_dict = {}

    Receptor_dict[0] = Receptor_1
    Receptor_dict[1] = Receptor_2
    Receptor_dict[2] = Receptor_3
    Receptor_dict[3] = Receptor_4

    num_files_Ligand = 5
    num_files_Receptor = 4

    X = []
    y = np.array(main['Class'])
    ids = np.array(main['IDs'])

    for i in range(len(y)):
        lst = []
        for j in range(num_files_Ligand):
            lst.append(Ligand_dict[j][i])
        for j in range(num_files_Receptor):
            lst.append(Receptor_dict[j][i])
        try:
            my_list = [ids[i], lst]
        except:
            print(len(ids), len(y), i, ids[i])
        X.append(my_list)

    if makeFolds:
        file_path = f'../all_data/raw_data/TrainDataUsable/Main/10Fold/fold_{fold_num}/'
    else:
        file_path = f'../all_data/raw_data/TrainDataUsable/Main/'
    pickleDump(file_path, f'{dataType}_{dataSubtype}_train_X.pkl', X)
    pickleDump(file_path, f'{dataType}_{dataSubtype}_train_y.pkl', y)

    X = np.array(X, dtype=object)
    X_copy = []
    for i in range(X.shape[0]):
        lst = []
        for j in range(len(X[i, 1])):
            lst.extend(X[i, 1][j][0])
        X_copy.append(lst)

    X_copy, y = smote.fit_resample(X_copy, y)

    X_up = []
    for i in range(len(X_copy)):
            lst = []
            for j in range(9):
                    tmp = np.array(X_copy[i][j * 128 : (j+1) * 128], dtype = np.float32)
                    lst.append(tmp.reshape(1, len(tmp)))
            X_up.append(lst)

    train_data = LigandReceptorDataset([['0']]* len(X_up), X_up, y) 

    if makeFolds:
        save_path = f'../all_data/processedData/MainModel/10Fold/{dataType}/{dataSubtype}/{fold_num}/'
    else:
        save_path = f'../all_data/processedData/MainModel/SingleDataset/{dataType}/{dataSubtype}/'
    pickleDump(save_path, 'train.pkl', train_data)


def processTestData(mode, dataType, dataSubtype, processDataType, processDataSubtype, fold_num, makeFolds):
    # dataType: testing data that we want to process, glass
    # subtype: all, human
    # processDataType: model that we are training, evolf
    # processDataSubType: all, human
    mypath = '../all_data/raw_data/TrainDataRaw/'

    if dataType == 'evolf':
        file_name = 'EvOlf'
    elif dataType == 'glass':
        file_name = 'GLASS'
    else:
        file_name = 'COMB'
    g2v = pd.read_csv(f'{mypath}25_{file_name}_Graph2Vec_Final.csv')
    sig = pd.read_csv(f'{mypath}25_{file_name}_Signaturizer_Final.csv')
    mord = pd.read_csv(f'{mypath}25_{file_name}_Mordred_Final.csv')
    m2v = pd.read_csv(f'{mypath}25_{file_name}_Mol2Vec_Final.csv')
    cb = pd.read_csv(f'{mypath}25_{file_name}_ChemBERTa_Final.csv')
    main = pd.read_csv(f'{mypath}25_{file_name}_MainDataset.csv')
    bfd = pd.read_csv(f'{mypath}25_{file_name}_ProtBFD_Final.csv')
    t5 = pd.read_csv(f'{mypath}25_{file_name}_ProtT5_Final.csv')
    mf = pd.read_csv(f'{mypath}25_{file_name}_MathFeature_Final.csv')
    r = pd.read_csv(f'{mypath}25_{file_name}_ProtR_Final.csv')
    
    sort_order = pd.Series(main.index, index=main['IDs']).to_dict()

    Ligand_1 = pd.merge(main, g2v, on='Ligand_ID', how='inner')
    Ligand_2 = pd.merge(main, sig, on='Ligand_ID', how='inner')
    Ligand_3 = pd.merge(main, mord, on='Ligand_ID', how='inner')
    Ligand_4 = pd.merge(main, m2v, on='Ligand_ID', how='inner')
    Ligand_5 = pd.merge(main, cb, on='Ligand_ID', how='inner')

    Receptor_1 = pd.merge(main, bfd, on='Receptor_ID', how='inner')
    Receptor_2 = pd.merge(main, t5, on='Receptor_ID', how='inner')
    Receptor_3 = pd.merge(main, mf, on='Receptor_ID', how='inner')
    Receptor_4 = pd.merge(main, r, on='Receptor_ID', how='inner')

    Ligand_1['sort_order'] = Ligand_1['IDs'].map(sort_order)
    Ligand_2['sort_order'] = Ligand_2['IDs'].map(sort_order)
    Ligand_3['sort_order'] = Ligand_3['IDs'].map(sort_order)
    Ligand_4['sort_order'] = Ligand_4['IDs'].map(sort_order)
    Ligand_5['sort_order'] = Ligand_5['IDs'].map(sort_order)

    Receptor_1['sort_order'] = Receptor_1['IDs'].map(sort_order)
    Receptor_2['sort_order'] = Receptor_2['IDs'].map(sort_order)
    Receptor_3['sort_order'] = Receptor_3['IDs'].map(sort_order)
    Receptor_4['sort_order'] = Receptor_4['IDs'].map(sort_order)

    Ligand_1.sort_values(by='sort_order', inplace=True)
    Ligand_1.drop(columns=['sort_order'], inplace=True)
    Ligand_1.reset_index(drop=True, inplace=True)

    Ligand_2.sort_values(by='sort_order', inplace=True)
    Ligand_2.drop(columns=['sort_order'], inplace=True)
    Ligand_2.reset_index(drop=True, inplace=True)

    Ligand_3.sort_values(by='sort_order', inplace=True)
    Ligand_3.drop(columns=['sort_order'], inplace=True)
    Ligand_3.reset_index(drop=True, inplace=True)

    Ligand_4.sort_values(by='sort_order', inplace=True)
    Ligand_4.drop(columns=['sort_order'], inplace=True)
    Ligand_4.reset_index(drop=True, inplace=True)

    Ligand_5.sort_values(by='sort_order', inplace=True)
    Ligand_5.drop(columns=['sort_order'], inplace=True)
    Ligand_5.reset_index(drop=True, inplace=True)


    Receptor_1.sort_values(by='sort_order', inplace=True)
    Receptor_1.drop(columns=['sort_order'], inplace=True)
    Receptor_1.reset_index(drop=True, inplace=True)

    Receptor_2.sort_values(by='sort_order', inplace=True)
    Receptor_2.drop(columns=['sort_order'], inplace=True)
    Receptor_2.reset_index(drop=True, inplace=True)

    Receptor_3.sort_values(by='sort_order', inplace=True)
    Receptor_3.drop(columns=['sort_order'], inplace=True)
    Receptor_3.reset_index(drop=True, inplace=True)

    Receptor_4.sort_values(by='sort_order', inplace=True)
    Receptor_4.drop(columns=['sort_order'], inplace=True)
    Receptor_4.reset_index(drop=True, inplace=True)
    
    
    if makeFolds:
        save_path = f'../all_data/processedData/MainModel/10Fold/fold_{fold_num}/ids/'
    else:
        save_path = f'../all_data/processedData/MainModel/SingleDataset/ids/'

    test_ids = pd.read_csv(save_path + f'{dataType}_{dataSubtype}_{mode}_ids.csv')
    sort_order = pd.Series(main.index, index=main['IDs']).to_dict()
    main = pd.merge(main, test_ids, on=['IDs'], how='inner')

    Ligand_1 = pd.merge(main, Ligand_1, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Ligand_2 = pd.merge(main, Ligand_2, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Ligand_3 = pd.merge(main, Ligand_3, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Ligand_4 = pd.merge(main, Ligand_4, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Ligand_5 = pd.merge(main, Ligand_5, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')

    Receptor_1 = pd.merge(main, Receptor_1, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Receptor_2 = pd.merge(main, Receptor_2, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Receptor_3 = pd.merge(main, Receptor_3, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Receptor_4 = pd.merge(main, Receptor_4, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')

    Ligand_1['sort_order'] = Ligand_1['IDs'].map(sort_order)
    Ligand_2['sort_order'] = Ligand_2['IDs'].map(sort_order)
    Ligand_3['sort_order'] = Ligand_3['IDs'].map(sort_order)
    Ligand_4['sort_order'] = Ligand_4['IDs'].map(sort_order)
    Ligand_5['sort_order'] = Ligand_5['IDs'].map(sort_order)

    Receptor_1['sort_order'] = Receptor_1['IDs'].map(sort_order)
    Receptor_2['sort_order'] = Receptor_2['IDs'].map(sort_order)
    Receptor_3['sort_order'] = Receptor_3['IDs'].map(sort_order)
    Receptor_4['sort_order'] = Receptor_4['IDs'].map(sort_order)

    Ligand_1.sort_values(by='sort_order', inplace=True)
    Ligand_1.drop(columns=['sort_order'], inplace=True)
    Ligand_1.reset_index(drop=True, inplace=True)

    Ligand_2.sort_values(by='sort_order', inplace=True)
    Ligand_2.drop(columns=['sort_order'], inplace=True)
    Ligand_2.reset_index(drop=True, inplace=True)

    Ligand_3.sort_values(by='sort_order', inplace=True)
    Ligand_3.drop(columns=['sort_order'], inplace=True)
    Ligand_3.reset_index(drop=True, inplace=True)

    Ligand_4.sort_values(by='sort_order', inplace=True)
    Ligand_4.drop(columns=['sort_order'], inplace=True)
    Ligand_4.reset_index(drop=True, inplace=True)

    Ligand_5.sort_values(by='sort_order', inplace=True)
    Ligand_5.drop(columns=['sort_order'], inplace=True)
    Ligand_5.reset_index(drop=True, inplace=True)


    Receptor_1.sort_values(by='sort_order', inplace=True)
    Receptor_1.drop(columns=['sort_order'], inplace=True)
    Receptor_1.reset_index(drop=True, inplace=True)

    Receptor_2.sort_values(by='sort_order', inplace=True)
    Receptor_2.drop(columns=['sort_order'], inplace=True)
    Receptor_2.reset_index(drop=True, inplace=True)

    Receptor_3.sort_values(by='sort_order', inplace=True)
    Receptor_3.drop(columns=['sort_order'], inplace=True)
    Receptor_3.reset_index(drop=True, inplace=True)

    Receptor_4.sort_values(by='sort_order', inplace=True)
    Receptor_4.drop(columns=['sort_order'], inplace=True)
    Receptor_4.reset_index(drop=True, inplace=True)


    Ligand_1 =  np.array(Ligand_1.iloc[:, 6:])
    Ligand_2 =  np.array(Ligand_2.iloc[:, 6:])
    Ligand_3 =  np.array(Ligand_3.iloc[:, 6:])
    Ligand_4 =  np.array(Ligand_4.iloc[:, 6:])
    Ligand_5 =  np.array(Ligand_5.iloc[:, 6:])

    Receptor_1 = np.array(Receptor_1.iloc[:, 5:])
    Receptor_2 = np.array(Receptor_2.iloc[:, 5:])
    Receptor_3 = np.array(Receptor_3.iloc[:, 5:])
    Receptor_4 = np.array(Receptor_4.iloc[:, 5:])

    if makeFolds:
        scaler_path = f'../all_data/processingModels/10Folds/fold_{fold_num}/scaler_models'
    else:
        scaler_path = '../all_data/processingModels/scaler_models'

    scaler_Ligand_1 = pickleRead(f'{scaler_path}/{processDataType}_{processDataSubtype}/', 'Ligand_1.pkl')
    Ligand_1 = scaler_Ligand_1.transform(Ligand_1)

    scaler_Ligand_2 = pickleRead(f'{scaler_path}/{processDataType}_{processDataSubtype}/', 'Ligand_2.pkl')
    Ligand_2 = scaler_Ligand_2.transform(Ligand_2)

    scaler_Ligand_3 = pickleRead(f'{scaler_path}/{processDataType}_{processDataSubtype}/', 'Ligand_3.pkl')
    Ligand_3 = scaler_Ligand_3.transform(Ligand_3)

    scaler_Ligand_4 = pickleRead(f'{scaler_path}/{processDataType}_{processDataSubtype}/', 'Ligand_4.pkl')
    Ligand_4 = scaler_Ligand_4.transform(Ligand_4)

    scaler_Ligand_5 = pickleRead(f'{scaler_path}/{processDataType}_{processDataSubtype}/', 'Ligand_5.pkl')
    Ligand_5 = scaler_Ligand_5.transform(Ligand_5)

    scaler_Receptor_1 = pickleRead(f'{scaler_path}/{processDataType}_{processDataSubtype}/', 'Receptor_1.pkl')
    Receptor_1 = scaler_Receptor_1.transform(Receptor_1)

    scaler_Receptor_2 = pickleRead(f'{scaler_path}/{processDataType}_{processDataSubtype}/', 'Receptor_2.pkl')
    Receptor_2 = scaler_Receptor_2.transform(Receptor_2)

    scaler_Receptor_3 = pickleRead(f'{scaler_path}/{processDataType}_{processDataSubtype}/', 'Receptor_3.pkl')
    Receptor_3 = scaler_Receptor_3.transform(Receptor_3)

    scaler_Receptor_4 = pickleRead(f'{scaler_path}/{processDataType}_{processDataSubtype}/', 'Receptor_4.pkl')
    Receptor_4 = scaler_Receptor_4.transform(Receptor_4)

    if makeFolds:
        pca_path = f'../all_data/processingModels/10Folds/fold_{fold_num}/pca_models'
    else:
        pca_path = '../all_data/processingModels/pca_models'

    pca_Ligand_1 = pickleRead(f'{pca_path}/{processDataType}_{processDataSubtype}/', 'Ligand_1.pkl')
    Ligand_1 = pca_Ligand_1.transform(Ligand_1)

    pca_Ligand_2 = pickleRead(f'{pca_path}/{processDataType}_{processDataSubtype}/', 'Ligand_2.pkl')
    Ligand_2 = pca_Ligand_2.transform(Ligand_2)

    pca_Ligand_3 = pickleRead(f'{pca_path}/{processDataType}_{processDataSubtype}/', 'Ligand_3.pkl')
    Ligand_3 = pca_Ligand_3.transform(Ligand_3)

    pca_Ligand_4 = pickleRead(f'{pca_path}/{processDataType}_{processDataSubtype}/', 'Ligand_4.pkl')
    Ligand_4 = pca_Ligand_4.transform(Ligand_4)

    pca_Ligand_5 = pickleRead(f'{pca_path}/{processDataType}_{processDataSubtype}/', 'Ligand_5.pkl')
    Ligand_5 = pca_Ligand_5.transform(Ligand_5)

    pca_Receptor_1 = pickleRead(f'{pca_path}/{processDataType}_{processDataSubtype}/', 'Receptor_1.pkl')
    Receptor_1 = pca_Receptor_1.transform(Receptor_1)

    pca_Receptor_2 = pickleRead(f'{pca_path}/{processDataType}_{processDataSubtype}/', 'Receptor_2.pkl')
    Receptor_2 = pca_Receptor_2.transform(Receptor_2)

    pca_Receptor_3 = pickleRead(f'{pca_path}/{processDataType}_{processDataSubtype}/', 'Receptor_3.pkl')
    Receptor_3 = pca_Receptor_3.transform(Receptor_3)

    pca_Receptor_4 = pickleRead(f'{pca_path}/{processDataType}_{processDataSubtype}/', 'Receptor_4.pkl')
    Receptor_4 = pca_Receptor_4.transform(Receptor_4)

    
    Ligand_1 = np.array(Ligand_1, dtype = np.float32)
    Ligand_2 = np.array(Ligand_2, dtype = np.float32)
    Ligand_3 = np.array(Ligand_3, dtype = np.float32)
    Ligand_4 = np.array(Ligand_4, dtype = np.float32)
    Ligand_5 = np.array(Ligand_5, dtype = np.float32)

    Receptor_1 = np.array(Receptor_1, dtype = np.float32)
    Receptor_2 = np.array(Receptor_2, dtype = np.float32)
    Receptor_3 = np.array(Receptor_3, dtype = np.float32)
    Receptor_4 = np.array(Receptor_4, dtype = np.float32)

    Ligand_1 = Ligand_1.reshape(Ligand_1.shape[0], 1, Ligand_1.shape[1])
    Ligand_2 = Ligand_2.reshape(Ligand_2.shape[0], 1, Ligand_2.shape[1])
    Ligand_3 = Ligand_3.reshape(Ligand_3.shape[0], 1, Ligand_3.shape[1])
    Ligand_4 = Ligand_4.reshape(Ligand_4.shape[0], 1, Ligand_4.shape[1])
    Ligand_5 = Ligand_5.reshape(Ligand_5.shape[0], 1, Ligand_5.shape[1])

    Receptor_1 = Receptor_1.reshape(Receptor_1.shape[0], 1, Receptor_1.shape[1])
    Receptor_2 = Receptor_2.reshape(Receptor_2.shape[0], 1, Receptor_2.shape[1])
    Receptor_3 = Receptor_3.reshape(Receptor_3.shape[0], 1, Receptor_3.shape[1])
    Receptor_4 = Receptor_4.reshape(Receptor_4.shape[0], 1, Receptor_4.shape[1])


    Ligand_dict = {}

    Ligand_dict[0] = Ligand_1
    Ligand_dict[1] = Ligand_2
    Ligand_dict[2] = Ligand_3
    Ligand_dict[3] = Ligand_4
    Ligand_dict[4] = Ligand_5

    Receptor_dict = {}

    Receptor_dict[0] = Receptor_1
    Receptor_dict[1] = Receptor_2
    Receptor_dict[2] = Receptor_3
    Receptor_dict[3] = Receptor_4

    num_files_Ligand = 5
    num_files_Receptor = 4

    X = []
    y = np.array(main['Class'])
    ids = np.array(main['IDs'])

    for i in range(len(y)):
        lst = []
        for j in range(num_files_Ligand):
            lst.append(Ligand_dict[j][i])
        for j in range(num_files_Receptor):
            lst.append(Receptor_dict[j][i])
        try:
            my_list = [ids[i], lst]
        except:
            print(len(ids), len(y), i, ids[i])
        X.append(my_list)

    if makeFolds:
        file_path = f'../all_data/raw_data/TrainDataUsable/Main/10Fold/fold_{fold_num}/'
    else:
        file_path = f'../all_data/raw_data/TrainDataUsable/Main/'
    pickleDump(file_path, f'{dataType}_{dataSubtype}_{mode}_X.pkl', X)
    pickleDump(file_path, f'{dataType}_{dataSubtype}_{mode}_y.pkl', y)


    data = LigandReceptorDataset(np.array(X)[:, 0].tolist(), np.array(X)[:, 1].tolist(), y) 

    if makeFolds:
        save_path = f'../all_data/processedData/MainModel/10Fold/{processDataType}/{processDataSubtype}/{fold_num}/'
    else:
        save_path = f'../all_data/processedData/MainModel/SingleDataset/{processDataType}/{processDataSubtype}/'
    pickleDump(save_path, f'{dataType}_{dataSubtype}_{mode}.pkl', data)



def processForMainMethod(makeFolds = False):

    total_folds = 1
    if makeFolds == True:
        total_folds = 10

    for fold_num in range(total_folds):

        fold_st = time.time()

        print(f" --- FOLD {fold_num} executing --- ")

        st = time.time()

        getIds(dataType = 'evolf', fold_num = fold_num, makeFolds = makeFolds)
        getIds(dataType = 'glass', fold_num = fold_num, makeFolds = makeFolds)
        getCombIds(fold_num, makeFolds)

        print(f'Time elapsed in processing IDs: {time.time() - st}')
        st = time.time()

        # -------------------------------------------------------------------------------------------

        processTrainData('evolf', 'all', fold_num, makeFolds)
        print(f'Time elapsed in making Evolf All train data: {time.time() - st}')
        st = time.time()

        processTrainData('evolf', 'human', fold_num, makeFolds)
        print(f'Time elapsed in making Evolf human train data: {time.time() - st}')
        st = time.time()

        processTrainData('glass', 'all', fold_num, makeFolds)
        print(f'Time elapsed in making Glass All train data: {time.time() - st}')
        st = time.time()

        processTrainData('glass', 'human', fold_num, makeFolds)
        print(f'Time elapsed in making Glass human train data: {time.time() - st}')
        st = time.time()

        processTrainData('comb', 'all', fold_num, makeFolds)
        print(f'Time elapsed in making Comb All train data: {time.time() - st}')
        st = time.time()

        processTrainData('comb', 'human', fold_num, makeFolds)
        print(f'Time elapsed in making Comb human train data: {time.time() - st}')
        st = time.time()

        processTestData('val', 'evolf', 'all', 'evolf', 'all', fold_num, makeFolds)
        print(f'Time elapsed in making Evolf All val data: {time.time() - st}')
        st = time.time()

        processTestData('val', 'evolf', 'human', 'evolf', 'human', fold_num, makeFolds)
        print(f'Time elapsed in making Evolf human val data: {time.time() - st}')
        st = time.time()

        processTestData('val', 'glass', 'all', 'glass', 'all', fold_num, makeFolds)
        print(f'Time elapsed in making glass All val data: {time.time() - st}')
        st = time.time()

        processTestData('val', 'glass', 'human', 'glass', 'human', fold_num, makeFolds)
        print(f'Time elapsed in making glass human val data: {time.time() - st}')
        st = time.time()

        processTestData('val', 'comb', 'all', 'comb', 'all', fold_num, makeFolds)
        print(f'Time elapsed in making comb All val data: {time.time() - st}')
        st = time.time()

        processTestData('val', 'comb', 'human', 'comb', 'human', fold_num, makeFolds)
        print(f'Time elapsed in making comb human val data: {time.time() - st}')
        st = time.time()

        processTestData('test', 'evolf', 'all', 'evolf', 'all', fold_num, makeFolds)
        processTestData('test', 'evolf', 'all', 'glass', 'all', fold_num, makeFolds)
        processTestData('test', 'evolf', 'all', 'comb', 'all', fold_num, makeFolds)
        processTestData('test', 'evolf', 'all', 'evolf', 'human', fold_num, makeFolds)
        processTestData('test', 'evolf', 'all', 'glass', 'human', fold_num, makeFolds)
        processTestData('test', 'evolf', 'all', 'comb', 'human', fold_num, makeFolds)
        print(f'Time elapsed in all evolf all test data: {time.time() - st}')
        st = time.time()

        processTestData('test', 'evolf', 'human', 'evolf', 'all', fold_num, makeFolds)
        processTestData('test', 'evolf', 'human', 'glass', 'all', fold_num, makeFolds)
        processTestData('test', 'evolf', 'human', 'comb', 'all', fold_num, makeFolds)
        processTestData('test', 'evolf', 'human', 'evolf', 'human', fold_num, makeFolds)
        processTestData('test', 'evolf', 'human', 'glass', 'human', fold_num, makeFolds)
        processTestData('test', 'evolf', 'human', 'comb', 'human', fold_num, makeFolds)
        print(f'Time elapsed in all evolf human test data: {time.time() - st}')
        st = time.time()

        processTestData('test', 'glass', 'all', 'evolf', 'all', fold_num, makeFolds)
        processTestData('test', 'glass', 'all', 'glass', 'all', fold_num, makeFolds)
        processTestData('test', 'glass', 'all', 'comb', 'all', fold_num, makeFolds)
        processTestData('test', 'glass', 'all', 'evolf', 'human', fold_num, makeFolds)
        processTestData('test', 'glass', 'all', 'glass', 'human', fold_num, makeFolds)
        processTestData('test', 'glass', 'all', 'comb', 'human', fold_num, makeFolds)
        print(f'Time elapsed in all glass all test data: {time.time() - st}')
        st = time.time()

        processTestData('test', 'glass', 'human', 'evolf', 'all', fold_num, makeFolds)
        processTestData('test', 'glass', 'human', 'glass', 'all', fold_num, makeFolds)
        processTestData('test', 'glass', 'human', 'comb', 'all', fold_num, makeFolds)
        processTestData('test', 'glass', 'human', 'evolf', 'human', fold_num, makeFolds)
        processTestData('test', 'glass', 'human', 'glass', 'human', fold_num, makeFolds)
        processTestData('test', 'glass', 'human', 'comb', 'human', fold_num, makeFolds)
        print(f'Time elapsed in all glass human test data: {time.time() - st}')
        st = time.time()

        processTestData('test', 'comb', 'all', 'evolf', 'all', fold_num, makeFolds)
        processTestData('test', 'comb', 'all', 'glass', 'all', fold_num, makeFolds)
        processTestData('test', 'comb', 'all', 'comb', 'all', fold_num, makeFolds)
        processTestData('test', 'comb', 'all', 'evolf', 'human', fold_num, makeFolds)
        processTestData('test', 'comb', 'all', 'glass', 'human', fold_num, makeFolds)
        processTestData('test', 'comb', 'all', 'comb', 'human', fold_num, makeFolds)
        print(f'Time elapsed in all comb all test data: {time.time() - st}')
        st = time.time()

        processTestData('test', 'comb', 'human', 'evolf', 'all', fold_num, makeFolds)
        processTestData('test', 'comb', 'human', 'glass', 'all', fold_num, makeFolds)
        processTestData('test', 'comb', 'human', 'comb', 'all', fold_num, makeFolds)
        processTestData('test', 'comb', 'human', 'evolf', 'human', fold_num, makeFolds)
        processTestData('test', 'comb', 'human', 'glass', 'human', fold_num, makeFolds)
        processTestData('test', 'comb', 'human', 'comb', 'human', fold_num, makeFolds)
        print(f'Time elapsed in all comb human test data: {time.time() - st}')
        st = time.time()

        

        print(f" --- FOLD {fold_num} completed in {fold_st - st} seconds --- ")
        st = time.time()


        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------


def makeFolds(makeFolds = True):
    total_folds = 10

    for fold_num in range(total_folds):

        fold_st = time.time()

        print(f" --- FOLD {fold_num} executing --- ")

        st = time.time()

        getIds(dataType = 'evolf', fold_num = fold_num, makeFolds = makeFolds)
        getIds(dataType = 'glass', fold_num = fold_num, makeFolds = makeFolds)
        getCombIds(fold_num, makeFolds)

        print(f'Time elapsed in processing IDs: {time.time() - st}')
        st = time.time()

        # -------------------------------------------------------------------------------------------

        # processTrainData('evolf', 'all', fold_num, makeFolds)
        # print(f'Time elapsed in making Evolf All train data: {time.time() - st}')
        # st = time.time()

        # processTrainData('evolf', 'human', fold_num, makeFolds)
        # print(f'Time elapsed in making Evolf human train data: {time.time() - st}')
        # st = time.time()

        # processTrainData('glass', 'all', fold_num, makeFolds)
        # print(f'Time elapsed in making Glass All train data: {time.time() - st}')
        # st = time.time()

        # processTrainData('glass', 'human', fold_num, makeFolds)
        # print(f'Time elapsed in making Glass human train data: {time.time() - st}')
        # st = time.time()

        processTrainData('comb', 'all', fold_num, makeFolds)
        print(f'Time elapsed in making Comb All train data: {time.time() - st}')
        st = time.time()

        # processTrainData('comb', 'human', fold_num, makeFolds)
        # print(f'Time elapsed in making Comb human train data: {time.time() - st}')
        # st = time.time()

        # processTestData('val', 'evolf', 'all', 'evolf', 'all', fold_num, makeFolds)
        # print(f'Time elapsed in making Evolf All val data: {time.time() - st}')
        # st = time.time()

        # processTestData('val', 'evolf', 'human', 'evolf', 'human', fold_num, makeFolds)
        # print(f'Time elapsed in making Evolf human val data: {time.time() - st}')
        # st = time.time()

        # processTestData('val', 'glass', 'all', 'glass', 'all', fold_num, makeFolds)
        # print(f'Time elapsed in making glass All val data: {time.time() - st}')
        # st = time.time()

        # processTestData('val', 'glass', 'human', 'glass', 'human', fold_num, makeFolds)
        # print(f'Time elapsed in making glass human val data: {time.time() - st}')
        # st = time.time()

        processTestData('val', 'comb', 'all', 'comb', 'all', fold_num, makeFolds)
        print(f'Time elapsed in making comb All val data: {time.time() - st}')
        st = time.time()

        # processTestData('val', 'comb', 'human', 'comb', 'human', fold_num, makeFolds)
        # print(f'Time elapsed in making comb human val data: {time.time() - st}')
        # st = time.time()

        # processTestData('test', 'evolf', 'all', 'evolf', 'all', fold_num, makeFolds)
        # processTestData('test', 'evolf', 'all', 'glass', 'all', fold_num, makeFolds)
        processTestData('test', 'evolf', 'all', 'comb', 'all', fold_num, makeFolds)
        # processTestData('test', 'evolf', 'all', 'evolf', 'human', fold_num, makeFolds)
        # processTestData('test', 'evolf', 'all', 'glass', 'human', fold_num, makeFolds)
        # processTestData('test', 'evolf', 'all', 'comb', 'human', fold_num, makeFolds)
        print(f'Time elapsed in all evolf all test data: {time.time() - st}')
        st = time.time()

        # processTestData('test', 'evolf', 'human', 'evolf', 'all', fold_num, makeFolds)
        # processTestData('test', 'evolf', 'human', 'glass', 'all', fold_num, makeFolds)
        processTestData('test', 'evolf', 'human', 'comb', 'all', fold_num, makeFolds)
        # processTestData('test', 'evolf', 'human', 'evolf', 'human', fold_num, makeFolds)
        # processTestData('test', 'evolf', 'human', 'glass', 'human', fold_num, makeFolds)
        # processTestData('test', 'evolf', 'human', 'comb', 'human', fold_num, makeFolds)
        print(f'Time elapsed in all evolf human test data: {time.time() - st}')
        st = time.time()

        # processTestData('test', 'glass', 'all', 'evolf', 'all', fold_num, makeFolds)
        # processTestData('test', 'glass', 'all', 'glass', 'all', fold_num, makeFolds)
        processTestData('test', 'glass', 'all', 'comb', 'all', fold_num, makeFolds)
        # processTestData('test', 'glass', 'all', 'evolf', 'human', fold_num, makeFolds)
        # processTestData('test', 'glass', 'all', 'glass', 'human', fold_num, makeFolds)
        # processTestData('test', 'glass', 'all', 'comb', 'human', fold_num, makeFolds)
        print(f'Time elapsed in all glass all test data: {time.time() - st}')
        st = time.time()

        # processTestData('test', 'glass', 'human', 'evolf', 'all', fold_num, makeFolds)
        # processTestData('test', 'glass', 'human', 'glass', 'all', fold_num, makeFolds)
        processTestData('test', 'glass', 'human', 'comb', 'all', fold_num, makeFolds)
        # processTestData('test', 'glass', 'human', 'evolf', 'human', fold_num, makeFolds)
        # processTestData('test', 'glass', 'human', 'glass', 'human', fold_num, makeFolds)
        # processTestData('test', 'glass', 'human', 'comb', 'human', fold_num, makeFolds)
        print(f'Time elapsed in all glass human test data: {time.time() - st}')
        st = time.time()

        # processTestData('test', 'comb', 'all', 'evolf', 'all', fold_num, makeFolds)
        # processTestData('test', 'comb', 'all', 'glass', 'all', fold_num, makeFolds)
        processTestData('test', 'comb', 'all', 'comb', 'all', fold_num, makeFolds)
        # processTestData('test', 'comb', 'all', 'evolf', 'human', fold_num, makeFolds)
        # processTestData('test', 'comb', 'all', 'glass', 'human', fold_num, makeFolds)
        # processTestData('test', 'comb', 'all', 'comb', 'human', fold_num, makeFolds)
        print(f'Time elapsed in all comb all test data: {time.time() - st}')
        st = time.time()

        # processTestData('test', 'comb', 'human', 'evolf', 'all', fold_num, makeFolds)
        # processTestData('test', 'comb', 'human', 'glass', 'all', fold_num, makeFolds)
        processTestData('test', 'comb', 'human', 'comb', 'all', fold_num, makeFolds)
        # processTestData('test', 'comb', 'human', 'evolf', 'human', fold_num, makeFolds)
        # processTestData('test', 'comb', 'human', 'glass', 'human', fold_num, makeFolds)
        # processTestData('test', 'comb', 'human', 'comb', 'human', fold_num, makeFolds)
        print(f'Time elapsed in all comb human test data: {time.time() - st}')
        st = time.time()

        

        print(f" --- FOLD {fold_num} completed in {st - fold_st} seconds --- ")
        st = time.time()


        # -------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------




def makeFullData():

    mypath = '../all_data/raw_data/TrainDataRaw/'
    dataType = 'comb'
    dataSubtype = 'all'
    

    if dataType == 'evolf':
        file_name = 'EvOlf'
    elif dataType == 'glass':
        file_name = 'GLASS'
    else:
        file_name = 'COMB'
    g2v = pd.read_csv(f'{mypath}25_{file_name}_Graph2Vec_Final.csv')
    sig = pd.read_csv(f'{mypath}25_{file_name}_Signaturizer_Final.csv')
    mord = pd.read_csv(f'{mypath}25_{file_name}_Mordred_Final.csv')
    m2v = pd.read_csv(f'{mypath}25_{file_name}_Mol2Vec_Final.csv')
    cb = pd.read_csv(f'{mypath}25_{file_name}_ChemBERTa_Final.csv')
    main = pd.read_csv(f'{mypath}25_{file_name}_MainDataset.csv')
    bfd = pd.read_csv(f'{mypath}25_{file_name}_ProtBFD_Final.csv')
    t5 = pd.read_csv(f'{mypath}25_{file_name}_ProtT5_Final.csv')
    mf = pd.read_csv(f'{mypath}25_{file_name}_MathFeature_Final.csv')
    r = pd.read_csv(f'{mypath}25_{file_name}_ProtR_Final.csv')

    
    sort_order = pd.Series(main.index, index=main['IDs']).to_dict()

    Ligand_1 = pd.merge(main, g2v, on='Ligand_ID', how='inner')
    Ligand_2 = pd.merge(main, sig, on='Ligand_ID', how='inner')
    Ligand_3 = pd.merge(main, mord, on='Ligand_ID', how='inner')
    Ligand_4 = pd.merge(main, m2v, on='Ligand_ID', how='inner')
    Ligand_5 = pd.merge(main, cb, on='Ligand_ID', how='inner')

    Receptor_1 = pd.merge(main, bfd, on='Receptor_ID', how='inner')
    Receptor_2 = pd.merge(main, t5, on='Receptor_ID', how='inner')
    Receptor_3 = pd.merge(main, mf, on='Receptor_ID', how='inner')
    Receptor_4 = pd.merge(main, r, on='Receptor_ID', how='inner')

    Ligand_1['sort_order'] = Ligand_1['IDs'].map(sort_order)
    Ligand_2['sort_order'] = Ligand_2['IDs'].map(sort_order)
    Ligand_3['sort_order'] = Ligand_3['IDs'].map(sort_order)
    Ligand_4['sort_order'] = Ligand_4['IDs'].map(sort_order)
    Ligand_5['sort_order'] = Ligand_5['IDs'].map(sort_order)

    Receptor_1['sort_order'] = Receptor_1['IDs'].map(sort_order)
    Receptor_2['sort_order'] = Receptor_2['IDs'].map(sort_order)
    Receptor_3['sort_order'] = Receptor_3['IDs'].map(sort_order)
    Receptor_4['sort_order'] = Receptor_4['IDs'].map(sort_order)

    Ligand_1.sort_values(by='sort_order', inplace=True)
    Ligand_1.drop(columns=['sort_order'], inplace=True)
    Ligand_1.reset_index(drop=True, inplace=True)

    Ligand_2.sort_values(by='sort_order', inplace=True)
    Ligand_2.drop(columns=['sort_order'], inplace=True)
    Ligand_2.reset_index(drop=True, inplace=True)

    Ligand_3.sort_values(by='sort_order', inplace=True)
    Ligand_3.drop(columns=['sort_order'], inplace=True)
    Ligand_3.reset_index(drop=True, inplace=True)

    Ligand_4.sort_values(by='sort_order', inplace=True)
    Ligand_4.drop(columns=['sort_order'], inplace=True)
    Ligand_4.reset_index(drop=True, inplace=True)

    Ligand_5.sort_values(by='sort_order', inplace=True)
    Ligand_5.drop(columns=['sort_order'], inplace=True)
    Ligand_5.reset_index(drop=True, inplace=True)


    Receptor_1.sort_values(by='sort_order', inplace=True)
    Receptor_1.drop(columns=['sort_order'], inplace=True)
    Receptor_1.reset_index(drop=True, inplace=True)

    Receptor_2.sort_values(by='sort_order', inplace=True)
    Receptor_2.drop(columns=['sort_order'], inplace=True)
    Receptor_2.reset_index(drop=True, inplace=True)

    Receptor_3.sort_values(by='sort_order', inplace=True)
    Receptor_3.drop(columns=['sort_order'], inplace=True)
    Receptor_3.reset_index(drop=True, inplace=True)

    Receptor_4.sort_values(by='sort_order', inplace=True)
    Receptor_4.drop(columns=['sort_order'], inplace=True)
    Receptor_4.reset_index(drop=True, inplace=True)
    
    save_path = f'../all_data/processedData/MainModel/SingleDataset/ids/'

    # train_ids = pd.read_csv(save_path + f'{dataType}_{dataSubtype}_train_ids.csv')
    # val_ids = pd.read_csv(save_path + f'{dataType}_{dataSubtype}_val_ids.csv')
    # test_ids = pd.read_csv(save_path + f'{dataType}_{dataSubtype}_test_ids.csv')
    
    sort_order = pd.Series(main.index, index=main['IDs']).to_dict()
    # main = pd.merge(main, train_ids, on=['IDs'], how='inner')

    Ligand_1 = pd.merge(main, Ligand_1, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Ligand_2 = pd.merge(main, Ligand_2, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Ligand_3 = pd.merge(main, Ligand_3, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Ligand_4 = pd.merge(main, Ligand_4, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Ligand_5 = pd.merge(main, Ligand_5, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')

    Receptor_1 = pd.merge(main, Receptor_1, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Receptor_2 = pd.merge(main, Receptor_2, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Receptor_3 = pd.merge(main, Receptor_3, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')
    Receptor_4 = pd.merge(main, Receptor_4, on=['IDs', 'Class', 'Ligand_ID', 'Receptor_ID', 'Species'], how='inner')

    Ligand_1['sort_order'] = Ligand_1['IDs'].map(sort_order)
    Ligand_2['sort_order'] = Ligand_2['IDs'].map(sort_order)
    Ligand_3['sort_order'] = Ligand_3['IDs'].map(sort_order)
    Ligand_4['sort_order'] = Ligand_4['IDs'].map(sort_order)
    Ligand_5['sort_order'] = Ligand_5['IDs'].map(sort_order)

    Receptor_1['sort_order'] = Receptor_1['IDs'].map(sort_order)
    Receptor_2['sort_order'] = Receptor_2['IDs'].map(sort_order)
    Receptor_3['sort_order'] = Receptor_3['IDs'].map(sort_order)
    Receptor_4['sort_order'] = Receptor_4['IDs'].map(sort_order)

    Ligand_1.sort_values(by='sort_order', inplace=True)
    Ligand_1.drop(columns=['sort_order'], inplace=True)
    Ligand_1.reset_index(drop=True, inplace=True)

    Ligand_2.sort_values(by='sort_order', inplace=True)
    Ligand_2.drop(columns=['sort_order'], inplace=True)
    Ligand_2.reset_index(drop=True, inplace=True)

    Ligand_3.sort_values(by='sort_order', inplace=True)
    Ligand_3.drop(columns=['sort_order'], inplace=True)
    Ligand_3.reset_index(drop=True, inplace=True)

    Ligand_4.sort_values(by='sort_order', inplace=True)
    Ligand_4.drop(columns=['sort_order'], inplace=True)
    Ligand_4.reset_index(drop=True, inplace=True)

    Ligand_5.sort_values(by='sort_order', inplace=True)
    Ligand_5.drop(columns=['sort_order'], inplace=True)
    Ligand_5.reset_index(drop=True, inplace=True)


    Receptor_1.sort_values(by='sort_order', inplace=True)
    Receptor_1.drop(columns=['sort_order'], inplace=True)
    Receptor_1.reset_index(drop=True, inplace=True)

    Receptor_2.sort_values(by='sort_order', inplace=True)
    Receptor_2.drop(columns=['sort_order'], inplace=True)
    Receptor_2.reset_index(drop=True, inplace=True)

    Receptor_3.sort_values(by='sort_order', inplace=True)
    Receptor_3.drop(columns=['sort_order'], inplace=True)
    Receptor_3.reset_index(drop=True, inplace=True)

    Receptor_4.sort_values(by='sort_order', inplace=True)
    Receptor_4.drop(columns=['sort_order'], inplace=True)
    Receptor_4.reset_index(drop=True, inplace=True)


    Ligand_1 =  np.array(Ligand_1.iloc[:, 6:])
    Ligand_2 =  np.array(Ligand_2.iloc[:, 6:])
    Ligand_3 =  np.array(Ligand_3.iloc[:, 6:])
    Ligand_4 =  np.array(Ligand_4.iloc[:, 6:])
    Ligand_5 =  np.array(Ligand_5.iloc[:, 6:])

    Receptor_1 = np.array(Receptor_1.iloc[:, 5:])
    Receptor_2 = np.array(Receptor_2.iloc[:, 5:])
    Receptor_3 = np.array(Receptor_3.iloc[:, 5:])
    Receptor_4 = np.array(Receptor_4.iloc[:, 5:])

    
    scaler_path = '../all_data/processingModels/scaler_models/full_data'

    scaler_Ligand_1 = StandardScaler()
    Ligand_1 = scaler_Ligand_1.fit_transform(Ligand_1)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Ligand_1.pkl', scaler_Ligand_1)

    scaler_Ligand_2 = StandardScaler()
    Ligand_2 = scaler_Ligand_2.fit_transform(Ligand_2)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Ligand_2.pkl', scaler_Ligand_2)

    scaler_Ligand_3 = StandardScaler()
    Ligand_3 = scaler_Ligand_3.fit_transform(Ligand_3)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Ligand_3.pkl', scaler_Ligand_3)

    scaler_Ligand_4 = StandardScaler()
    Ligand_4 = scaler_Ligand_4.fit_transform(Ligand_4)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Ligand_4.pkl', scaler_Ligand_4)

    scaler_Ligand_5 = StandardScaler()
    Ligand_5 = scaler_Ligand_5.fit_transform(Ligand_5)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Ligand_5.pkl', scaler_Ligand_5)


    scaler_Receptor_1 = StandardScaler()
    Receptor_1 = scaler_Receptor_1.fit_transform(Receptor_1)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Receptor_1.pkl', scaler_Receptor_1)

    scaler_Receptor_2 = StandardScaler()
    Receptor_2 = scaler_Receptor_2.fit_transform(Receptor_2)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Receptor_2.pkl', scaler_Receptor_2)

    scaler_Receptor_3 = StandardScaler()
    Receptor_3 = scaler_Receptor_3.fit_transform(Receptor_3)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Receptor_3.pkl', scaler_Receptor_3)

    scaler_Receptor_4 = StandardScaler()
    Receptor_4 = scaler_Receptor_4.fit_transform(Receptor_4)
    pickleDump(f'{scaler_path}/{dataType}_{dataSubtype}/', 'Receptor_4.pkl', scaler_Receptor_4)

    pca_path = '../all_data/processingModels/pca_models/full_data'

    pca_Ligand_1 = IncrementalPCA(n_components=128)
    Ligand_1 = pca_Ligand_1.fit_transform(Ligand_1)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Ligand_1.pkl', pca_Ligand_1)

    pca_Ligand_2 = IncrementalPCA(n_components=128)
    Ligand_2 = pca_Ligand_2.fit_transform(Ligand_2)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Ligand_2.pkl', pca_Ligand_2)


    pca_Ligand_3 = IncrementalPCA(n_components=128)
    Ligand_3 = pca_Ligand_3.fit_transform(Ligand_3)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Ligand_3.pkl', pca_Ligand_3)


    pca_Ligand_4 = IncrementalPCA(n_components=128)
    Ligand_4 = pca_Ligand_4.fit_transform(Ligand_4)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Ligand_4.pkl', pca_Ligand_4)


    pca_Ligand_5 = IncrementalPCA(n_components=128)
    Ligand_5 = pca_Ligand_5.fit_transform(Ligand_5)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Ligand_5.pkl', pca_Ligand_5)


    pca_Receptor_1 = IncrementalPCA(n_components=128)
    Receptor_1 = pca_Receptor_1.fit_transform(Receptor_1)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Receptor_1.pkl', pca_Receptor_1)


    pca_Receptor_2 = IncrementalPCA(n_components=128)
    Receptor_2 = pca_Receptor_2.fit_transform(Receptor_2)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Receptor_2.pkl', pca_Receptor_2)

    pca_Receptor_3 = IncrementalPCA(n_components=128)
    Receptor_3 = pca_Receptor_3.fit_transform(Receptor_3)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Receptor_3.pkl', pca_Receptor_3)

    pca_Receptor_4 = IncrementalPCA(n_components=128)
    Receptor_4 = pca_Receptor_4.fit_transform(Receptor_4)
    pickleDump(f'{pca_path}/{dataType}_{dataSubtype}/', 'Receptor_4.pkl', pca_Receptor_4)

    
    Ligand_1 = np.array(Ligand_1, dtype = np.float32)
    Ligand_2 = np.array(Ligand_2, dtype = np.float32)
    Ligand_3 = np.array(Ligand_3, dtype = np.float32)
    Ligand_4 = np.array(Ligand_4, dtype = np.float32)
    Ligand_5 = np.array(Ligand_5, dtype = np.float32)

    Receptor_1 = np.array(Receptor_1, dtype = np.float32)
    Receptor_2 = np.array(Receptor_2, dtype = np.float32)
    Receptor_3 = np.array(Receptor_3, dtype = np.float32)
    Receptor_4 = np.array(Receptor_4, dtype = np.float32)

    Ligand_1 = Ligand_1.reshape(Ligand_1.shape[0], 1, Ligand_1.shape[1])
    Ligand_2 = Ligand_2.reshape(Ligand_2.shape[0], 1, Ligand_2.shape[1])
    Ligand_3 = Ligand_3.reshape(Ligand_3.shape[0], 1, Ligand_3.shape[1])
    Ligand_4 = Ligand_4.reshape(Ligand_4.shape[0], 1, Ligand_4.shape[1])
    Ligand_5 = Ligand_5.reshape(Ligand_5.shape[0], 1, Ligand_5.shape[1])

    Receptor_1 = Receptor_1.reshape(Receptor_1.shape[0], 1, Receptor_1.shape[1])
    Receptor_2 = Receptor_2.reshape(Receptor_2.shape[0], 1, Receptor_2.shape[1])
    Receptor_3 = Receptor_3.reshape(Receptor_3.shape[0], 1, Receptor_3.shape[1])
    Receptor_4 = Receptor_4.reshape(Receptor_4.shape[0], 1, Receptor_4.shape[1])


    Ligand_dict = {}

    Ligand_dict[0] = Ligand_1
    Ligand_dict[1] = Ligand_2
    Ligand_dict[2] = Ligand_3
    Ligand_dict[3] = Ligand_4
    Ligand_dict[4] = Ligand_5

    Receptor_dict = {}

    Receptor_dict[0] = Receptor_1
    Receptor_dict[1] = Receptor_2
    Receptor_dict[2] = Receptor_3
    Receptor_dict[3] = Receptor_4

    num_files_Ligand = 5
    num_files_Receptor = 4

    X = []
    y = np.array(main['Class'])
    ids = np.array(main['IDs'])

    for i in range(len(y)):
        lst = []
        for j in range(num_files_Ligand):
            lst.append(Ligand_dict[j][i])
        for j in range(num_files_Receptor):
            lst.append(Receptor_dict[j][i])
        try:
            my_list = [ids[i], lst]
        except:
            print(len(ids), len(y), i, ids[i])
        X.append(my_list)

    file_path = f'../all_data/raw_data/TrainDataUsable/Main/full_data/'
    pickleDump(file_path, f'{dataType}_{dataSubtype}_full_X.pkl', X)
    pickleDump(file_path, f'{dataType}_{dataSubtype}_full_y.pkl', y)

    X = np.array(X, dtype=object)
    X_copy = []
    for i in range(X.shape[0]):
        lst = []
        for j in range(len(X[i, 1])):
            lst.extend(X[i, 1][j][0])
        X_copy.append(lst)

    X_copy, y = smote.fit_resample(X_copy, y)

    X_up = []
    for i in range(len(X_copy)):
            lst = []
            for j in range(9):
                    tmp = np.array(X_copy[i][j * 128 : (j+1) * 128], dtype = np.float32)
                    lst.append(tmp.reshape(1, len(tmp)))
            X_up.append(lst)

    full_data = LigandReceptorDataset([['0']]* len(X_up), X_up, y) 

    save_path = f'../all_data/processedData/MainModel/Full_data/'
    pickleDump(save_path, 'full_data.pkl', full_data)

    # file_path = '../all_data/raw_data/TrainDataUsable/main/'

    # all_X = pickleRead(file_path, 'comb_all_X.pkl')
    # all_y = pickleRead(file_path, 'comb_all_y.pkl')

    # all_X_copy = []

    # all_X = np.array(all_X, dtype=object)
    # for i in range(all_X.shape[0]):
    #         lst = []
    #         for j in range(len(all_X[i, 1])):
    #                 lst.extend(all_X[i, 1][j][0])
    #         all_X_copy.append(lst)

    # all_X_copy, all_y = smote.fit_resample(all_X_copy, all_y)


    # all_X_up = []

    # for i in range(len(all_X_copy)):
    #         lst = []
    #         for j in range(9):
    #                 tmp = np.array(all_X_copy[i][j * 128 : (j+1) * 128], dtype = np.float32)
    #                 lst.append(tmp.reshape(1, len(tmp)))
    #         all_X_up.append(lst)

    # for i in range(len(all_X_up)):
    #     if len(all_X_up[i]) != 9:
    #         print(i)
    # full_data = LigandReceptorDataset([['0']]* len(all_X_up), all_X_up, all_y) 
    # pickleDump('../all_data/processedData/FullData/', 'full_data.pkl', full_data)







if __name__ == '__main__':
    # processForMainMethod(makeFolds=False)
    # makeFolds()
    makeFullData()