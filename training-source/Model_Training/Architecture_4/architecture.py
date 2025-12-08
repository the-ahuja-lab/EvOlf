from myImports import *
import torch.nn.init as init

class MyModel(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=128, num_heads=4, num_layers=2):
        super(MyModel, self).__init__()

        self.key_transformer = nn.Transformer(d_model=input_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first = True)
        self.key_cnn = nn.Conv1d(in_channels = 5, out_channels = 100, kernel_size = 2)
        self.key_lstm = nn.LSTM(input_size = 127, hidden_size = 128, num_layers = 2, batch_first = True, dropout = 0.2)

        self.lock_transformer = nn.Transformer(d_model=input_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first = True)
        self.lock_cnn = nn.Conv1d(in_channels = 4, out_channels = 100, kernel_size = 2)
        self.lock_lstm = nn.LSTM(input_size = 127, hidden_size = 128, num_layers = 2, batch_first = True, dropout = 0.2)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), 
                                        nn.Linear(64, 16), nn.ReLU(), 
                                        nn.Linear(16, 2))
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights for linear layers using Xavier (Glorot) initialization
        for layer in [self.classifier[0], self.classifier[2], self.classifier[4]]:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0)  # Initialize bias to zeros

        # Initialize LSTM weights
        for layer in [self.key_lstm, self.lock_lstm]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    init.constant_(param, 0)

    
    def forward(self, k1, k2, k3, k4, k5, l1, l2, l3, l4):

        key_ft = torch.cat([k1, k2, k3, k4, k5], dim=1)
        key_ft = self.key_transformer(key_ft, key_ft)
        key_ft_norm = self.layer_norm1(key_ft)  
        key_ft = self.key_cnn(key_ft + key_ft_norm)
        key_ft, (h_k, c_k) = self.key_lstm(key_ft)
        key_ft_final = h_k[-1]

        lock_ft = torch.cat([l1, l2, l3, l4], dim=1)
        lock_ft = self.lock_transformer(lock_ft, lock_ft)
        lock_ft_norm = self.layer_norm2(lock_ft)  
        lock_ft = self.lock_cnn(lock_ft + lock_ft_norm)
        lock_ft, (h_l, c_l) = self.lock_lstm(lock_ft)
        lock_ft_final = h_l[-1]
        
        concat = torch.stack([key_ft_final, lock_ft_final], dim=1)
        concat_final = torch.mean(concat, dim=1)
        output = self.dropout(concat_final)
        output = self.classifier(output)
        return output, key_ft_final, lock_ft_final, concat_final