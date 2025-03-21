import ast
import json
import os
import random
import time
import uuid

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from agents.ensemble_agent.helpers import encode_board, cell_to_index, format_time, save_list, load_list, \
    data_generator, list_to_key, key_to_list

from agents.ensemble_agent.ChessHybridCNN import ChessCNN

PAWN = 1
KNIGHT = 3
BISHOP = 3.5
ROOK = 5
QUEEN = 9
KING = 10
NONE = 0

WHITE = 1
BLACK = -1

WIN_COUNT = 0
LOSS_COUNT = 1
OCC = 2

translator = {'p': PAWN, 'n': KNIGHT, 'b': BISHOP, 'q': QUEEN, 'k': KING, 'r': ROOK, '.': NONE}

middle_files = 'cdef'
middle_ranks = '3456'

white_back_files = 'abcdefgh'
white_back_ranks = '12'

black_back_files = 'abcdefgh'
black_back_ranks = '78'

#v6 model architecture


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

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epoch_num, margin=0.05, save_model=False):
    model.to(device)
    for _ in range(1):
        if train_loader is not None:
            model.train()
            running_loss = 0.0
            l = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).float()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                if l == 0:
                    print("Initial loss BCe: ", running_loss / len(inputs))
                    l = 1
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Training Loss BCE: {epoch_loss:.8f}")
            if save_model:
                model.save_model(f"./{epoch_num}_cnn.pth")
        if valid_loader is not None:
            model.eval()
            total = 0
            total_correct_5 = 0
            total_correct_10 = 0
            total_correct_25 = 0
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
                        total += 1
                        if closely_equals(p, t, margin=0.05):
                            total_correct_5 += 1
                        if closely_equals(p, t, margin=0.1):
                            total_correct_10 += 1
                        if closely_equals(p, t, margin=0.25):
                            total_correct_25 += 1
            print("Sub Accuracy: ", total_correct_5 / total, total_correct_10 / total, total_correct_25 / total)
            return total, total_correct_5, total_correct_10, total_correct_25


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
                    try:
                        encoded_board = encode_board(current_board)
                        bpc, wpc, bqc, wqc, brc, wrc, bbc, wbc, bnc, wnc, bpwc, wpwc = get_point_counts(encoded_board)
                        div = ((bpc + wpc+1))
                        bpc /= div
                        wpc /= div
                        div = ((bqc + wqc+1))
                        bqc /= div
                        wqc /= div
                        div = ((brc + wrc+1))
                        brc /= div
                        wrc /= div
                        div = ((bbc + wbc+1))
                        bbc /= div
                        wbc /= div
                        div = ((bnc + wnc+1))
                        bnc /= div
                        wnc /= div
                        div = ((bpwc + wpwc+1))
                        bpwc /= div
                        wpwc /= div
                        bcim, wcim = get_middle_counts(encoded_board)
                        bcbr, wcbr = get_back_row_counts(encoded_board)
                        div = ((bcim + wcim+1)* 4)
                        bcim /= div
                        wcim /= div
                        div = ((bcbr + wcbr+1) * 4)
                        bcbr /= div
                        wcbr /= div
                        sub_x = [0.0] * 64
                        for row in encoded_board:
                            for p, cell in row:
                                index = cell_to_index(cell)
                                value = translator.get(p.lower())
                                if p.lower() == p:
                                    value *= BLACK
                                sub_x[index] = value / 20
                        sub_x += [bpc, wpc, bqc, wqc, brc, wrc, bbc, wbc, bnc, wnc, bpwc, wpwc, bcim, wcim, bcbr, wcbr]
                        key_x = list_to_key(sub_x)
                        if tracker.get(key_x) is None:
                            tracker[key_x] = "0:0:0"
                        sub_label = tracker[key_x].split(":")
                        sub_label[WIN_COUNT] = str(int(sub_label[WIN_COUNT]) + label[WIN_COUNT])
                        sub_label[LOSS_COUNT] = str(int(sub_label[LOSS_COUNT]) + label[LOSS_COUNT])
                        sub_label[OCC] = str(int(sub_label[OCC]) + 1)
                        tracker[key_x] = ':'.join(sub_label)
                        current_board.push_san(move)
                    except Exception as e:
                        break
    all_states = list(tracker.keys())
    random.shuffle(all_states)
    number_of_states_for_each_occ_count = {}
    for state in all_states:
        LABEL = tracker.get(state).split(":")
        LABEL[WIN_COUNT] = int(LABEL[WIN_COUNT])
        LABEL[LOSS_COUNT] = int(LABEL[LOSS_COUNT])
        LABEL[OCC] = int(LABEL[OCC])

        if number_of_states_for_each_occ_count.get(LABEL[OCC]) is None:
            number_of_states_for_each_occ_count[LABEL[OCC]] = 0
        number_of_states_for_each_occ_count[LABEL[OCC]] += 1

        label_sum = LABEL[WIN_COUNT] + LABEL[LOSS_COUNT]
        label = [LABEL[WIN_COUNT] / label_sum, LABEL[LOSS_COUNT] / label_sum]

        lstate = key_to_list(state)
        x.append(lstate)
        y.append(label)
        del tracker[state]
        if len(x) == 1500000:
            save_list(f"./objects/{str(uuid.uuid4())}.pkl", [x, y])
            x = []
            y = []
    save_list(f"./objects/{str(uuid.uuid4())}.pkl", [x, y])
    plt.clf()
    plt.plot(list(number_of_states_for_each_occ_count.keys()), list(number_of_states_for_each_occ_count.values()), marker='o', markersize=2, linestyle='None')
    plt.grid()
    plt.title("Occurrence count vs States Count")
    plt.xlabel("Occurrence count")
    plt.ylabel("States Count")
    plt.savefig("./graphs/state_occurrence_distribution.png", format="png")


if __name__ == "__main__":

    seed = 904056181
    generator = torch.Generator().manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get_data()

    mode = "train"

    file_list = os.listdir("./objects")

    model = ChessCNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000025)
    num_epochs = list(range(100))

    x = []
    y5 = []
    y10 = []
    y25 = []

    for epoch in num_epochs:
        files_processed = 0
        print("Starting training")
        random.shuffle(file_list)
        total = 0
        total_correct_5 = 0
        total_correct_10 = 0
        total_correct_25 = 0
        if mode == "test":
            model = ChessCNN.load_model(f"./{epoch+1}_cnn.pth")
        for chunk_data in data_generator(file_list, batch_size=32, file_prefix="./objects/"):
            inputs, outputs = chunk_data
            input_tensor = torch.tensor(inputs, dtype=torch.float32)
            output_tensor = torch.tensor(outputs, dtype=torch.float32)
            dataset = TensorDataset(input_tensor, output_tensor)
            train_size = int(0.9 * len(dataset))
            test_size = len(dataset) - train_size
            print("Splitting Data . . .")
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
            train_loader = DataLoader(train_dataset, batch_size=2000, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=2000, shuffle=False)
            if mode == "train":
                train_model(model, train_loader, None, criterion, optimizer, device=device, epoch_num=epoch+1, save_model=files_processed==len(file_list) - 1)
            if mode == "test":
                t, tc5, tc10, tc25 = train_model(model, None, test_loader, criterion, optimizer, device=device, epoch_num=epoch+1, margin=0.5)
                total += t
                total_correct_5 += tc5
                total_correct_10 += tc10
            files_processed += 1
        if mode == "test":
            x.append(epoch)
            y5.append(total_correct_5 / total)
            y10.append(total_correct_10 / total)
            y25.append(total_correct_25 / total)
    if mode == "test":
        plt.clf()
        plt.plot(x, y5, label="margin=0.05")
        plt.plot(x, y10, label="margin=0.1")
        plt.plot(x, y25, label="margin=0.25")
        plt.grid()
        plt.legend()
        plt.ylim(0, 1)
        plt.title("New Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig("./graphs/model_accuracy.png", format="png")

