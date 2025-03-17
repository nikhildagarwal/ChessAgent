import random
import json
import chess

import numpy as np

import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, TensorDataset

from train import encode_board, cell_to_index

from train import get_data as get_data_with_duplicates

from NeuralNetworkAttention import ModelAttention

import matplotlib.pyplot as plt

import os


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


def get_data():
    tracker = {}
    print("Getting Data")
    counter = 0
    stats = {}
    ranges_data = {}
    for wi in range(1, 17):
        for wj in range(1, 17):
            ranges_data[(wi, wj)] = {'x': [], 'y': []}
    for filename in os.listdir("../data"):
        filepath = f"../data/{filename}"
        with open(filepath, "r") as json_file:
            print("Opening: ", filepath)
            data = json.load(json_file)
            for game in data['games']:
                move_count = 0
                board = chess.Board()
                sub_data = data['games'].get(game)
                count_white = sub_data['count_white']
                count_black = sub_data['count_black']
                for i, move in enumerate(sub_data['moves']):
                    plus_tracker  = 0
                    move_count += 1
                    arr = encode_board(board)
                    piece_map = board.piece_map()
                    white_count = sum(1 for piece in piece_map.values() if piece.color == chess.WHITE)
                    black_count = sum(1 for piece in piece_map.values() if piece.color == chess.BLACK)
                    move_str = str(board.push_san(move))
                    init_cell = move_str[0:2]
                    dest_cell = move_str[2:]
                    action_player_is_white = i % 2 == 0
                    sub_x = [0.0] * 65
                    sub_x[64] = float(int(action_player_is_white))
                    for row in arr:
                        for p, cell in row:
                            index = cell_to_index(cell)
                            value = translator.get(p.lower())
                            if p.lower() == p:
                                value *= BLACK
                            sub_x[index] = value
                    temp_x = tuple(sub_x)
                    sub_y = np.array([0.0] * 64)
                    sub_y[cell_to_index(init_cell)] = 1.0
                    sub_y[cell_to_index(dest_cell)] = 1.0
                    if action_player_is_white:
                        if count_white:
                            counter += 1
                            if tracker.get(temp_x) is None:
                                tracker[temp_x] = [np.array([0.0] * 64), 0 + plus_tracker, white_count, black_count]
                            tracker[temp_x][0] += sub_y
                            tracker[temp_x][1] += 1
                    else:
                        if count_black:
                            counter += 1
                            if tracker.get(temp_x) is None:
                                tracker[temp_x] = [np.array([0.0] * 64), 0 + plus_tracker, white_count, black_count]
                            tracker[temp_x][0] += sub_y
                            tracker[temp_x][1] += 1
                if stats.get(move_count) is None:
                    stats[move_count] = 0
                stats[move_count] += 1
    x = []
    y = []
    for state in tracker:
        tracker[state][0] /= (np.sum(tracker[state][0]))
        wc = tracker[state][2]
        bc = tracker[state][3]
        for _ in range(tracker[state][1]):
            ranges_data[(wc, bc)]['x'].append(state)
            ranges_data[(wc, bc)]['y'].append(tracker[state][0])
            x.append(list(state))
            y.append(tracker[state][0].tolist())
    print("Number of states: ", counter)
    print(stats)
    """plt.plot(list(stats.keys()), list(stats.values()), marker='o', linestyle='None')
    plt.show()"""
    return x, y


def test_model(model, dataloader, model_name, mode="Test"):
    """
    Evaluates the model on the provided dataloader by computing the mean squared error (MSE)
    between the output probability distributions and the label distributions.

    Assumes that the model outputs log probabilities (via log_softmax) and the labels are probability distributions.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.MSELoss()  # Uses the default 'mean' reduction

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            outputs, _ = model(batch_inputs)
            probs = torch.exp(outputs)
            loss = criterion(probs, batch_labels)
            total_loss += loss.item() * batch_labels.size(0)
            total_samples += batch_labels.size(0)

    overall_mse = total_loss / total_samples
    print(f"{mode} {model_name} MSE: {overall_mse:.20f}")
    return overall_mse




if __name__ == "__main__":
    seed = 904056181
    generator = torch.Generator().manual_seed(seed)
    inputs, outputs = get_data()
    input_tensor = torch.tensor(inputs, dtype=torch.float32)
    output_tensor = torch.tensor(outputs, dtype=torch.float32)
    dataset = TensorDataset(input_tensor, output_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, generator=generator)
    model_paths = os.listdir('./models')
    try:
        model_paths.sort(key=lambda x: int(x.split("_")[0]))
    except ValueError:
        pass
    x = []
    mse = []
    for i, model_path in enumerate(model_paths):
        model = ModelAttention.load_model("./models/"+model_path)
        t_mse = test_model(model, test_loader, model_path)
        x.append(i)
        mse.append(t_mse)
    plt.clf()
    plt.plot(x, mse)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("MSE per epoch (863,796 samples)")
    plt.savefig('./data/mse.png', format='png')


