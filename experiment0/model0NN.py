import torch
import torch.nn as nn
import torch.nn.functional as F

class Model0NN(nn.Module):
    def __init__(self):
        super(Model0NN, self).__init__()
        self.layer1 = nn.Linear(68, 130)
        self.layer2 = nn.Linear(130, 390)
        self.layer3 = nn.Linear(390, 800)
        self.layer4 = nn.Linear(800, 400)
        self.layer5 = nn.Linear(400, 350)
        self.layer6 = nn.Linear(350, 300)
        self.layer7 = nn.Linear(300, 250)
        self.layer8 = nn.Linear(250, 200)
        self.layer9 = nn.Linear(200, 150)
        self.layer10 = nn.Linear(150, 100)
        self.layer11 = nn.Linear(100, 64)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        x = torch.sigmoid(self.layer4(x))
        x = torch.sigmoid(self.layer5(x))
        x = torch.sigmoid(self.layer6(x))
        x = torch.sigmoid(self.layer7(x))
        x = torch.sigmoid(self.layer8(x))
        x = torch.sigmoid(self.layer9(x))
        x = torch.sigmoid(self.layer10(x))
        x = self.layer11(x)
        x = F.log_softmax(x, dim=1)
        return x

    def save_model(self, filepath):
        """
        Saves the model's state dictionary to the given filepath.
        """
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """
        Loads the model's state dictionary from the given filepath
        and returns an instance of Model0NN.
        """
        model = cls()
        model.load_state_dict(torch.load(filepath))
        model.eval()
        return model

