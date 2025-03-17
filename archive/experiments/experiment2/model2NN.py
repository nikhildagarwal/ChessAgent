import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelRNN(nn.Module):
    def __init__(self, embed_dim=64, num_layers=1, nonlinearity='tanh'):
        super(ModelRNN, self).__init__()
        self.rnn = nn.RNN(input_size=64, hidden_size=embed_dim, num_layers=num_layers,
                          nonlinearity=nonlinearity, batch_first=True)
        self.layer1 = nn.Linear(embed_dim + 1, 130)
        self.layer2 = nn.Linear(130, 390)
        self.layer3 = nn.Linear(390, 200)
        self.layer4 = nn.Linear(200, 150)
        self.layer5 = nn.Linear(150, 100)
        self.layer6 = nn.Linear(100, 64)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Split the input into main features and the extra binary flag.
        main_features = x[..., :64]  # shape: (batch, seq_len, 64)
        binary_features = x[..., 64]  # shape: (batch, seq_len)
        out, hidden = self.rnn(main_features)
        context = hidden[-1]
        extra_feature = binary_features[:, -1].unsqueeze(1)
        context = torch.cat([context, extra_feature], dim=1)
        x = torch.sigmoid(self.layer1(context))
        x = torch.sigmoid(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        x = torch.sigmoid(self.layer4(x))
        x = torch.sigmoid(self.layer5(x))
        x = self.layer6(x)
        x = F.log_softmax(x, dim=1)
        return x, None

    def save_model(self, filepath):
        """
        Saves the model's state dictionary to the given filepath.
        """
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, embed_dim=64, num_layers=1, nonlinearity='tanh'):
        """
        Loads the model's state dictionary from the given filepath and returns an instance.
        """
        model = cls(embed_dim=embed_dim, num_layers=num_layers, nonlinearity=nonlinearity)
        model.load_state_dict(torch.load(filepath))
        model.eval()  # Set to evaluation mode.
        return model
