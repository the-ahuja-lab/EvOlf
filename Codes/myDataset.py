from torch.utils.data import Dataset
import copy

class LigandReceptorDataset(Dataset):
  def __init__(self, ids, X, y):
    self.ids = ids
    self.X = X
    self.y = y
    
    self.n_samples = len(X)

  def __getitem__(self, index):
    ids, data, label = self.ids[index], self.X[index], self.y[index]
    return ids, data, label

  def __len__(self):
    return self.n_samples

  def merge(self, obj):
    new_ids = copy.deepcopy(self.ids)
    new_X = copy.deepcopy(self.X)
    new_Y = copy.deepcopy(self.y)

    for i in range(len(obj)):
      ids, a,b = obj[i]
      new_ids.append(ids)
      new_X.append(a)
      new_Y.append(b)

    return LigandReceptorDataset(new_ids, new_X, new_Y)



class CustomDataset(Dataset):
  def __init__(self, ids, X, y):
    self.ids = ids
    self.X = X
    self.y = y
    
    self.n_samples = len(X)

  def __getitem__(self, index):
    ids, data, label = self.ids[index], self.X[index], self.y[index]
    return ids, data, label

  def __len__(self):
    return self.n_samples

  def merge(self, obj):

    new_ids = self.ids
    new_X = self.X
    new_Y = self.y

    for i in range(len(obj)):
      ids, a,b = obj[i]
      new_ids.append(ids)
      new_X.append(a)
      new_Y.append(b)

    return CustomDataset(new_ids, new_X, new_Y)