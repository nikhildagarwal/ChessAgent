import json
import os
import time

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from agents.ensemble_agent.helpers import encode_board, cell_to_index, format_time, save_list

PAWN = 1
KNIGHT = 3
BISHOP = 3.5
ROOK = 5
QUEEN = 9
KING = 10
NONE = 0

WHITE = 1
BLACK = -1

translator = {'p': PAWN, 'n': KNIGHT, 'b': BISHOP, 'q': QUEEN, 'k': KING, 'r': ROOK, '.': NONE}

middle_files = 'cdef'
middle_ranks = '3456'

white_back_files = 'abcdefgh'
white_back_ranks = '12'

black_back_files = 'abcdefgh'
black_back_ranks = '78'

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

def convert_list_for_acc_check(arr, l1, l2):
    temp = []
    for sub_arr in arr:
        if l1 < sub_arr[0] <l2 and l1 < sub_arr[1] <l2:
            temp.append([0.5, 0.5])
        else:
            if sub_arr[0] >= l2:
                temp.append([1,0])
            else:
                temp.append([0,1])
    return temp

def closely_equals(pred, target, margin=0.05):
    if pred[0] - margin < target[0] <= pred[0] + margin:
        return True
    return False

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=1, margin=0.05):
    model.to(device)
    st = time.time()
    for epoch in range(num_epochs):
        if train_loader is not None:
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).float()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss BCE: {epoch_loss:.8f}")
            et = time.time()
            print(format_time(et - st))
            if epoch % 2 == 0:
                model.save_model(f"./{epoch}_cnn.pth")
        if valid_loader is not None:
            model.eval()
            total = 0
            total_correct = 0
            unq_states = set()
            total_count = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device).float()
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    probs = probs.tolist()
                    targets = targets.tolist()
                    inputs = inputs.tolist()
                    for p, t, inp in zip(probs, targets, inputs):
                        state = tuple(inp)
                        total_count += 1
                        if closely_equals(p, t, margin=margin):
                            total += 1
                            if state not in unq_states:
                                total_correct += 1
                        unq_states.add(state)
            unq_accuracy = total_correct / len(unq_states)
            print(len(unq_states))
            accuracy = total / total_count
            print(f"Epoch {epoch + 1}/{num_epochs} - Accuracy: {accuracy:.8f}, {unq_accuracy:.8f}")
            return accuracy, unq_accuracy


def get_point_counts(encoded_board):
    bpc, wpc, bqc, wqc, brc, wrc, bbc, wbc, bnc, wnc, bpwc, wpwc= 0,0,0,0,0,0,0,0,0,0,0,0
    for row in encoded_board:
        for item in row:
            p, cell = item
            if p.lower() != "." and p.lower() != 'k':
                if p.lower() == p: # piece is black
                    bpc += translator.get(p.lower())
                    if p.lower() == 'q':
                        bqc += 1
                    if p.lower() == 'r':
                        brc += 1
                    if p.lower() == 'b':
                        bbc += 1
                    if p.lower() == 'n':
                        bnc += 1
                    if p.lower() == 'p':
                        bpwc += 1
                else: # piece is white
                    wpc += translator.get(p.lower())
                    if p.lower() == 'q':
                        wqc += 1
                    if p.lower() == 'r':
                        wrc += 1
                    if p.lower() == 'b':
                        wbc += 1
                    if p.lower() == 'n':
                        wnc += 1
                    if p.lower() == 'p':
                        wpwc += 1
    return bpc, wpc, bqc, wqc, brc, wrc, bbc, wbc, bnc, wnc, bpwc, wpwc


def get_middle_counts(encoded_board):
    bcim, wcim = 0, 0
    for row in encoded_board:
        for item in row:
            p, cell = item
            file, rank = cell[0], cell[1]
            if file in middle_files and rank in middle_ranks:
                if p.lower() != ".":
                    if p.lower() == p: # black
                        bcim += 1
                    else: # white
                        wcim += 1
    return bcim, wcim


def get_back_row_counts(encoded_board):
    bcbr, wcbr = 0, 0
    for row in encoded_board:
        for item in row:
            p, cell = item
            file, rank = cell[0], cell[1]
            if file in black_back_files and rank in black_back_ranks:
                if p.lower() == p:
                    bcbr += 1
            if file in white_back_files and rank in white_back_ranks:
                if p.upper() == p:
                    wcbr += 1
    return bcbr, wcbr

def get_data():
    x = []
    y = []
    tracker = {}
    for path in os.listdir("data"):
        with open("./data/"+path, 'r') as file:
            data = json.load(file)
            print(f"Parsing ../data/{path} -- Games: {data['number_of_games']}")
            for game_uid in data["games"]:
                current_board = chess.Board()
                label = [int(data["games"][game_uid]["count_white"]), int(data["games"][game_uid]["count_black"])]
                for move in data["games"][game_uid]["moves"]:
                    encoded_board = encode_board(current_board)
                    bpc, wpc, bqc, wqc, brc, wrc, bbc, wbc, bnc, wnc, bpwc, wpwc = get_point_counts(encoded_board)
                    div = ((bpc + wpc+1) * 16)
                    bpc /= div
                    wpc /= div
                    div = ((bqc + wqc+1) * 16)
                    bqc /= div
                    wqc /= div
                    div = ((brc + wrc+1) * 16)
                    brc /= div
                    wrc /= div
                    div = ((bbc + wbc+1) * 16)
                    bbc /= div
                    wbc /= div
                    div = ((bnc + wnc+1) * 16)
                    bnc /= div
                    wnc /= div
                    div = ((bpwc + wpwc+1) * 16)
                    bpwc /= div
                    wpwc /= div
                    bcim, wcim = get_middle_counts(encoded_board)
                    bcbr, wcbr = get_back_row_counts(encoded_board)
                    div = ((bcim + wcim+1) * 16)
                    bcim /= div
                    wcim /= div
                    div = ((bcbr + wcbr+1) * 16)
                    bcbr /= div
                    wcbr /= div
                    sub_x = [0.0] * 64
                    for row in encoded_board:
                        for p, cell in row:
                            index = cell_to_index(cell)
                            value = translator.get(p.lower())
                            if p.lower() == p:
                                value *= BLACK
                            sub_x[index] = value / 10
                    sub_x += [bpc, wpc, bqc, wqc, brc, wrc, bbc, wbc, bnc, wnc, bpwc, wpwc, bcim, wcim, bcbr, wcbr]
                    sub_x = tuple(sub_x)
                    if tracker.get(sub_x) is None:
                        tracker[sub_x] = {'win_count': 0, 'lose_count': 0, 'occ': 0}
                    tracker[sub_x]['win_count'] += label[0]
                    tracker[sub_x]['lose_count'] += label[1]
                    tracker[sub_x]['occ'] += 1
                    current_board.push_san(move)
    for state in tracker:
        label_sum = tracker[state]['win_count'] + tracker[state]['lose_count']
        label = [tracker[state]['win_count'] / label_sum, tracker[state]['lose_count'] / label_sum]
        lstate = list(state)
        x.append(lstate)
        y.append(label)
    return x, y


if __name__ == "__main__":
    seed = 904056181
    generator = torch.Generator().manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unique_states = get_data()

    save_list("./objects/unique_states_1.pkl", unique_states)

    exit(0)

    print("Number of unique states: ", len(inputs))

    input_tensor = torch.tensor(inputs, dtype=torch.float32)

    output_tensor = torch.tensor(outputs, dtype=torch.float32)

    dataset = TensorDataset(input_tensor, output_tensor)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    print("Splitting Data . . .")
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, generator=generator)

    print("Starting Training . . .")
    # Assuming ChessCNN is defined as in the previous snippet
    model = ChessCNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, train_loader, None, criterion, optimizer, device=device, num_epochs=101)
    """for mg in [0.05, 0.1, 0.2, 0.25]:
        x = []
        y = []
        uy = []
        for i in range(101):
            if i%2 == 0:
                model = ChessCNN.load_model(f"./{i}_cnn.pth")
                acc, unq_acc = train_model(model, None, test_loader, criterion, optimizer, device, margin=mg)
                x.append(i)
                y.append(acc)
                uy.append(unq_acc)
        plt.plot(x, y, label=f"margin = {mg}")
        plt.plot(x, y, label=f"margin = {mg}, unq")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid()
    plt.title("epoch vs accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("./graphs/unique_never_seen_v1.png", format="png")"""

