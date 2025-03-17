import json
import os
import random
import time

import chess

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from experiment0.model0NN import Model0NN

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


def train_model(model, dataloader, criterion, optimizer, num_epochs, start_epoch=0, clip_value=1.0):
    epochs = []
    losses = []
    st = time.time()
    # Loop from the start epoch up to the total number of epochs
    for epoch in range(start_epoch, num_epochs+1):
        running_loss = 0.0
        for batch_inputs, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            # Directly use the multi-label targets (assumed to be a 64-element multi-hot vector)
            loss = criterion(outputs, batch_labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        epochs.append(epoch + 1)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if epoch % 100 == 0:
            model.save_model(f"./models/{epoch}_model0nn.pth")
            print("Execution Time: ", format_time(time.time() - st))

    print("Training finished.")
    tracker = {'epochs': epochs, 'losses': losses}
    with open("data/training_loss.json", "w") as json_file:
        json.dump(tracker, json_file, indent=4)



def encode_board(board):
    board_array = []
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            piece_symbol = piece.symbol() if piece is not None else "."
            cell = (chr(file + ord('a')) + str(rank + 1))
            row.append((piece_symbol, cell))
        board_array.append(row)
    return board_array


def cell_to_index(cell: str) -> int:
    """
    Converts a chess board cell (e.g., 'a3', 'c6') into a square index (0 to 63).
    The indexing is done such that:
      - 'a1' maps to 0
      - 'b1' maps to 1
      - ...
      - 'h1' maps to 7
      - 'a2' maps to 8, and so on.
    """
    file_index = ord(cell[0].lower()) - ord('a')
    rank_index = int(cell[1]) - 1
    index = rank_index * 8 + file_index
    return index


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
                    move_count += 1
                    arr = encode_board(board)
                    piece_map = board.piece_map()
                    white_count = sum(1 for piece in piece_map.values() if piece.color == chess.WHITE)
                    black_count = sum(1 for piece in piece_map.values() if piece.color == chess.BLACK)
                    move_str = str(board.push_san(move))
                    init_cell = move_str[0:2]
                    dest_cell = move_str[2:]
                    action_player_is_white = i % 2 == 0
                    sub_x = [0.0] * 68
                    sub_x[64] = float(int(action_player_is_white))
                    sub_x[65] = white_count + black_count
                    sub_x[66] = white_count
                    sub_x[67] = black_count
                    for row in arr:
                        for p, cell in row:
                            index = cell_to_index(cell)
                            value = translator.get(p.lower())
                            if p.lower() == p:
                                value *= BLACK
                            sub_x[index] = value / 10
                    temp_x = tuple(sub_x)
                    sub_y = np.array([0.0] * 64)
                    sub_y[cell_to_index(init_cell)] = 1.0
                    sub_y[cell_to_index(dest_cell)] = 1.0
                    if action_player_is_white:
                        if count_white:
                            counter += 1
                            if tracker.get(temp_x) is None:
                                tracker[temp_x] = [np.array([0.0] * 64), 0, white_count, black_count]
                            tracker[temp_x][0] += sub_y
                            tracker[temp_x][1] += 1
                    else:
                        if count_black:
                            counter += 1
                            if tracker.get(temp_x) is None:
                                tracker[temp_x] = [np.array([0.0] * 64), 0, white_count, black_count]
                            tracker[temp_x][0] += sub_y
                            tracker[temp_x][1] += 1
                if stats.get(move_count) is None:
                    stats[move_count] = 0
                stats[move_count] += 1
    x = []
    y = []
    for state in tracker:
        tracker[state][0] /= (np.sum(tracker[state][0]))
        for _ in range(tracker[state][1]):
            x.append(state)
            y.append(tracker[state][0])
    print("Number of states: ", len(x))
    print(stats)
    """plt.plot(list(stats.keys()), list(stats.values()), marker='o', linestyle='None')
    plt.show()"""
    return x, y


def format_time(seconds):
    """
    Convert a time given in seconds to a formatted string HH:MM:SS.

    Args:
        seconds (int or float): The time duration in seconds.

    Returns:
        str: The formatted time string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    seed = 904056181
    generator = torch.Generator().manual_seed(seed)

    inputs, outputs = get_data()

    input_tensor = torch.tensor(inputs, dtype=torch.float32)

    output_tensor = torch.tensor(outputs, dtype=torch.float32)

    dataset = TensorDataset(input_tensor, output_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Optionally, create DataLoaders for training/testing
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, generator=generator)

    model = Model0NN()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, train_loader, criterion, optimizer, num_epochs=10001)

    with open("data/training_loss.json", "r") as json_file:
        data = json.load(json_file)
        x = data['epochs']
        y = data['losses']
        plt.plot(x, y)
        plt.xlabel('epoch')
        plt.ylabel('average loss')
        plt.title('epoch vs loss')
        plt.savefig('./data/loss_plot.png', format='png')



    """model = Model0NN.load_model("./model0nn.pth")

    test_model(model, test_loader)"""