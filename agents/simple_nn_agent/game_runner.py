from models.NeuralNetworkAttention import ModelAttention
from helpers import *
import torch

BOARD = chess.Board()
MODEL = ModelAttention.load_model("models/1300_model0nn.pth")

move_count = 0

state_dict = {'white': 'player', 'black': 'model'}

translator = {0: 'white', 1: 'black'}

def run_model_move():
    input("Enter for model move\n")
    encoded_board = encode_board(BOARD)
    input_tensor = get_input_tensor(encoded_board, is_white_turn)
    output = MODEL.forward(input_tensor)
    probs = torch.exp(output[0])
    probs_list = probs.tolist()[0]
    src_cells, dst_cells = parse_cells(probs_list, is_white_turn, BOARD)
    sorted_legal_moves = get_sorted_legal_moves(src_cells, dst_cells, BOARD)
    chosen_move = choose_move(sorted_legal_moves)
    make_move(chosen_move, BOARD)
    print(f"MODEL-{translator[move_count % 2]} moved from {chosen_move[0:2]} to {chosen_move[2:]}")

def run_player_move():
    valid_input = False
    while not valid_input:
        input_move = input("Please enter a move\n")
        input_move = input_move.lower()
        is_valid_move = check_valid_move(input_move, BOARD)
        if is_valid_move:
            make_move(input_move, BOARD)
            valid_input = True
            print(f"Player moved from {input_move[0:2]} to {input_move[2:]}")

while not BOARD.is_game_over():
    is_white_turn = move_count % 2 == 0
    if is_white_turn:
        if state_dict['white'] == 'model':
            run_model_move()
        else:
            run_player_move()
    else:
        if state_dict['black'] == 'player':
            run_player_move()
        else:
            run_model_move()
    move_count += 1

outcome = BOARD.outcome()  # Retrieves the game outcome
if outcome.winner is None:
    print("The game is a draw.")
elif outcome.winner:
    print("White wins!")
else:
    print("Black wins!")


