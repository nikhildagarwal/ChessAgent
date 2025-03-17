import json
import os
import random

import chess
import numpy as np
import torch

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

def get_input_tensor(encoded_board, is_white_turn):
    sub_x = [0.0] * 65
    sub_x[64] = float(int(is_white_turn))
    for row in encoded_board:
        for p, cell in row:
            index = cell_to_index(cell)
            value = translator.get(p.lower())
            if p.lower() == p:
                value *= BLACK
            sub_x[index] = value
    return torch.tensor([sub_x], dtype=torch.float32)

def parse_cells(probs_list, is_white_turn, board):
    src_cells = {}
    dst_cells = {}
    for i in range(len(probs_list)):
        cell = index_to_cell(i)
        square = chess.parse_square(cell)
        piece = board.piece_at(square)
        if piece is not None: # if it is my piece
            my_piece = piece.color == is_white_turn
            if my_piece:
                src_cells[cell] = probs_list[i]
            else:
                dst_cells[cell] = probs_list[i]
        else: # if it is an open cell
            dst_cells[cell] = probs_list[i]
    return src_cells, dst_cells

def get_sorted_legal_moves(src_cells, dst_cells, board: chess.Board):
    legal_moves = []
    total = 0
    for src_cell in src_cells:
        for dst_cell in dst_cells:
            move_str = src_cell+dst_cell
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                val = src_cells[src_cell] * dst_cells[dst_cell]
                legal_moves.append([val, move_str])
                total += val
    legal_moves.sort(key=lambda x: x[0], reverse=True)
    for i in range(len(legal_moves)):
        legal_moves[i][0] /= total
    return legal_moves

def check_valid_move(move_str, board: chess.Board):
    try:
        move = chess.Move.from_uci(move_str)
        return move in board.legal_moves
    except Exception as e:
        return False

def make_move(move_str, board: chess.Board):
    move = chess.Move.from_uci(move_str)
    board.push(move)


def cell_to_index(cell: str) -> int:
    file_index = ord(cell[0].lower()) - ord('a')
    rank_index = int(cell[1]) - 1
    index = rank_index * 8 + file_index
    return index

def index_to_cell(index: int) -> str:
    rank_index = index // 8
    file_index = index % 8
    cell = chr(file_index + ord('a')) + str(rank_index + 1)
    return cell

def choose_move(sorted_moves):
    weights = [item[0] for item in sorted_moves]
    move_strings = [item[1] for item in sorted_moves]
    selected_move = random.choices(move_strings, weights=weights, k=1)[0]
    return selected_move