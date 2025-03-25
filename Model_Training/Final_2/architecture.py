from myImports import *
import torch.nn.init as init

class MyModel(nn.Module):
  def __init__(self, input_dim=128, hidden_dim=128, num_heads=4, num_layers=2):
      
      super(MyModel, self).__init__()

      self.key_transformer = nn.Transformer(d_model=input_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first=True)
      self.key_cnn = nn.Sequential(
          nn.Conv1d(5, 1, kernel_size=2, groups=1),
          nn.ReLU(),
      )

      self.lock_transformer = nn.Transformer(d_model=input_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first=True)
      self.lock_cnn = nn.Sequential(
          nn.Conv1d(4, 1, kernel_size=2, groups=1),
          nn.ReLU(),
      )

      self.layer_norm1 = nn.LayerNorm(hidden_dim)
      self.layer_norm2 = nn.LayerNorm(hidden_dim)

      self.dropout = nn.Dropout(0.4)
      self.classifier = nn.Sequential(
          nn.Linear(127, 64),
          nn.ReLU(),
          nn.Linear(64, 16),
          nn.ReLU(),
          nn.Linear(16, 2)
      )

      self.initialize_weights()

  def initialize_weights(self):
      for module in self.modules():
          if isinstance(module, (nn.Linear, nn.Conv1d)):
              init.xavier_uniform_(module.weight)
              if module.bias is not None:
                  init.constant_(module.bias, 0)
          elif isinstance(module, nn.LayerNorm):
              init.normal_(module.weight, 1.0, 0.02)
              init.constant_(module.bias, 0)

    
  def forward(self, k1, k2, k3, k4, k5, l1, l2, l3, l4):

    key_ft = torch.cat([k1, k2, k3, k4, k5], dim=1)
    key_ft = self.key_transformer(key_ft, key_ft)
    key_ft_norm = self.layer_norm1(key_ft)  
    key_ft_final = self.key_cnn(key_ft + key_ft_norm)

    lock_ft = torch.cat([l1, l2, l3, l4], dim=1)
    lock_ft = self.lock_transformer(lock_ft, lock_ft)
    lock_ft_norm = self.layer_norm2(lock_ft)  
    lock_ft_final = self.lock_cnn(lock_ft + lock_ft_norm)
    
    concat = torch.cat([key_ft_final, lock_ft_final], dim=1)
    # print(concat.shape)
    concat_final = torch.mean(concat, dim=1)
    # print(concat_final.shape)
    output = self.dropout(concat_final)
    output = self.classifier(output)
    return output, key_ft_final, lock_ft_final, concat_final