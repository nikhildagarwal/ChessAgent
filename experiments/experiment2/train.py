import json
import random

import chess

import matplotlib.pyplot as plt

from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from experiment2.model2NN import ModelRNN

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

import time
import torch


def train_model(model, dataloader, criterion, optimizer, num_epochs, start_epoch=0, clip_value=1.0, device='cpu'):
    epochs = []
    losses = []
    st = time.time()
    # Move the model to the desired device (CPU or GPU)
    model.to(device)
    model.train()
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        for batch_inputs, batch_labels in dataloader:
            # Move data to the device
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            # Unpack the output; we only need the logits for loss calculation.
            outputs, _ = model(batch_inputs)
            loss = criterion(outputs, batch_labels.float())
            loss.backward()
            # Clip gradients if needed
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
    with open("./data/training_loss.json", "w") as json_file:
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
    with open("../data/kasparov.json", "r") as json_file:
        data = json.load(json_file)
        x = []
        y = []
        for game in data['games']:
            board = chess.Board()
            sub_data = data['games'].get(game)
            count_white = sub_data['count_white']
            count_black = sub_data['count_black']
            for i, move in enumerate(sub_data['moves']):
                move_str = str(board.push_san(move))
                init_cell = move_str[0:2]
                dest_cell = move_str[2:]
                action_player_is_white = i % 2 == 0
                arr = encode_board(board)
                sub_x = [0.0] * 65
                sub_x[64] = float(int(action_player_is_white))
                for row in arr:
                    for p, cell in row:
                        index = cell_to_index(cell)
                        value = translator.get(p.lower())
                        if p.lower() == p:
                            value *= BLACK
                        sub_x[index] = value
                sub_y = [0.0] * 64
                sub_y[cell_to_index(init_cell)] = 1.0
                sub_y[cell_to_index(dest_cell)] = 1.0
                if action_player_is_white:
                    if count_white:
                        x.append(sub_x)
                        y.append(sub_y)
                else:
                    if count_black:
                        x.append(sub_x)
                        y.append(sub_y)
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
    random.seed(904056181)

    inputs, outputs = get_data()

    input_tensor = torch.tensor(inputs, dtype=torch.float32)

    output_tensor = torch.tensor(outputs, dtype=torch.float32)

    dataset = TensorDataset(input_tensor, output_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Optionally, create DataLoaders for training/testing
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = ModelRNN()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, train_loader, criterion, optimizer, num_epochs=501)

    with open("./data/training_loss.json", "r") as json_file:
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