import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv1_1 = nn.Conv1d(32,32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm1d(32)
        self.conv1_2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv2_1 = nn.Conv1d(64, 64, kernel_size=4, padding=1)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.conv2_2 = nn.Conv1d(64, 128, kernel_size=4, padding=1)
        self.bn2_2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=7, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.conv6 = nn.Conv1d(512, 512, kernel_size=9, padding=1)
        self.bn6 = nn.BatchNorm1d(512)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.small_pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.dropout_conv = nn.Dropout(0.02)
        self.small_dropout_conv = nn.Dropout(0.005)

        self.fc1 = nn.Linear(1536, 256*4)
        self.bn_fc1 = nn.BatchNorm1d(256*4)
        self.dropout_fc1 = nn.Dropout(0.005)
        self.fc2 = nn.Linear(256*4, 256*2)
        self.bn_fc2 = nn.BatchNorm1d(256*2)
        self.dropout_fc2 = nn.Dropout(0.005)
        self.fc3 = nn.Linear(256*2, 256)
        self.bn_fc3 = nn.BatchNorm1d(256)
        self.dropout_fc3 = nn.Dropout(0.005)
        self.fc4 = nn.Linear(256, 100)
        self.bn_fc4 = nn.BatchNorm1d(100)


        self.conv7 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(8)
        self.conv8 = nn.Conv1d(8, 8, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(8)


        self.dropout_fc4 = nn.Dropout(0.005)
        self.fc5 = nn.Linear(400, 200)
        self.bn_fc5 = nn.BatchNorm1d(200)
        self.dropout_fc5 = nn.Dropout(0.005)
        self.fc5_1 = nn.Linear(200, 100)
        self.bn_fc5_1 = nn.BatchNorm1d(100)
        self.dropout_fc5_1 = nn.Dropout(0.005)
        self.fc5_2 = nn.Linear(100, 50)
        self.bn_fc5_2 = nn.BatchNorm1d(50)
        self.dropout_fc5_2 = nn.Dropout(0.005)
        self.fc6 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout_fc3(x)
        x = F.relu(self.bn_fc4(self.fc4(x)))
        x = self.dropout_fc4(x)

        x = x.unsqueeze(1)
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool(x)
        x = self.small_dropout_conv(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc5(self.fc5(x)))
        x = self.dropout_fc5(x)
        x = F.relu(self.bn_fc5_1(self.fc5_1(x)))
        x = self.dropout_fc5_1(x)
        x = F.relu(self.bn_fc5_2(self.fc5_2(x)))
        x = self.dropout_fc5_2(x)
        x = self.fc6(x)
        return x

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, to_train=False):
        model = cls()
        model.load_state_dict(torch.load(filepath))
        model.eval()
        if to_train:
            model.train()
        return model


# Example usage:
if __name__ == "__main__":
    # Create a random input with batch size 10.
    x = torch.randn(10, 80)
    model = ChessCNN()
    output = model(x)
    print("Output shape:", output.shape)