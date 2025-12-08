from myImports import *
import torch.nn.init as init

class MyModel(nn.Module):
  def __init__(self):
        
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4)

        self.lstm1 = nn.LSTM(input_size=124, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=125, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
          nn.Linear(256, 64),
          nn.ReLU(),
          nn.Linear(64, 16),
          nn.ReLU(),
          nn.Linear(16, 2)
      )

        # Initialize weights
        self.initialize_weights()

  def initialize_weights(self):
      for layer in [self.conv1, self.conv2]:
          if isinstance(layer, (nn.Conv2d, nn.Linear)):
              init.xavier_uniform_(layer.weight)  # Xavier initialization for weights
              if layer.bias is not None:
                  init.constant_(layer.bias, 0)  # Initialize bias to zeros

      # Initialize LSTM weights
      for layer in [self.lstm1, self.lstm2]:
          for name, param in layer.named_parameters():
              if 'weight' in name:
                  init.orthogonal_(param)  # Orthogonal initialization for LSTM weights
              elif 'bias' in name:
                  init.constant_(param, 0)  # Initialize bias to zeros

    
  def forward(self, k1, k2, k3, k4, k5, l1, l2, l3, l4):

    key_ft = torch.cat([k1, k2, k3, k4, k5], dim=1)
    key_ft = key_ft.reshape(key_ft.shape[0], 1, key_ft.shape[1], key_ft.shape[2])
    key_ft = self.conv1(key_ft)
    key_ft = key_ft.reshape(key_ft.shape[0], key_ft.shape[1], key_ft.shape[3])
    # print(key_ft.shape)

    lock_ft = torch.cat([l1, l2, l3, l4], dim=1)
    lock_ft = lock_ft.reshape(lock_ft.shape[0], 1, lock_ft.shape[1], lock_ft.shape[2])
    lock_ft = self.conv2(lock_ft)
    lock_ft = lock_ft.reshape(lock_ft.shape[0], lock_ft.shape[1], lock_ft.shape[3])
    # print(lock_ft.shape)

    key_ft, (h_k, c_k) = self.lstm1(key_ft)
    lock_ft, (h_l, c_l) = self.lstm2(lock_ft)

    key_ft_final = h_k[-1]
    lock_ft_final = h_l[-1]
    
    concat_final = torch.cat([key_ft_final, lock_ft_final], dim = 1)
    # print(concat_final.shape)
    output = self.dropout(concat_final)
    output = self.classifier(output)

    return output, key_ft_final, lock_ft_final, concat_final
    