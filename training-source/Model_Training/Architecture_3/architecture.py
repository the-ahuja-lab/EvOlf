from torch import nn
import torch.nn.init as init
import torch

class TransformerLSTMFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_hidden_dim, num_heads, num_layers):
        super(TransformerLSTMFusion, self).__init__()

        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )

        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, hidden_dim)

        # Weight initialization for LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                init.constant_(param, 0.0)

        # Weight initialization for Linear layer
        init.xavier_uniform_(self.fc.weight)
        init.constant_(self.fc.bias, 0.0)

    def forward(self, input_vectors):
        transformer_output = self.transformer(input_vectors, input_vectors)
        lstm_output, _ = self.lstm(transformer_output)
        lstm_output = lstm_output[:, -1, :]
        fused_vector = self.fc(lstm_output)

        return fused_vector


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, stride=1, padding=1),  # Adjust parameters as needed
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
                
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )


        self.keyFusion = TransformerLSTMFusion(input_dim=128, hidden_dim=128, lstm_hidden_dim=128, num_heads=4,
                                               num_layers=2)
        self.lockFusion = TransformerLSTMFusion(input_dim=128, hidden_dim=128, lstm_hidden_dim=128, num_heads=4,
                                                num_layers=2)


        # Initialize Convolutional Layers
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0.0)

        # Initialize Classifier Layers
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0.0)

    def forward(self, k1, k2, k3, k4, k5, l1, l2, l3, l4):
        key_ft = torch.cat([k1, k2, k3, k4, k5], dim=1)
        key_ft_final = self.keyFusion(key_ft)
        # print(key_ft_final)

        lock_ft = torch.cat([l1, l2, l3, l4], dim=1)
        lock_ft_final = self.lockFusion(lock_ft)
        # print(lock_ft_final)

        concat = torch.stack([key_ft_final, lock_ft_final], dim=1)
        concat_final = torch.mean(concat, dim=1)
        # print(concat_final)

        cnn_input = torch.cat([concat_final.unsqueeze(1), key_ft_final.unsqueeze(1), lock_ft_final.unsqueeze(1)], dim=1)
        cnn_output = self.conv_layers(cnn_input)
        # print(cnn_input.shape)

        classifier_input = cnn_output.view(-1, 64 * 32)
        output = self.classifier(classifier_input)

        return output, key_ft_final, lock_ft_final, concat_final
