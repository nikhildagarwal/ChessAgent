from helpers import *

BOARD = chess.Board()

move_count = 0

state_dict = {'white': 'model', 'black': 'player'}

translator = {0: 'white', 1: 'black'}

unique_states = get_unique_state_data()

while not BOARD.is_game_over():
    is_white_turn = move_count % 2 == 0
    if is_white_turn and state_dict['white'] == 'model':
        # if True:
        input("Enter for Move\n")
        encoded_board = encode_board(BOARD)
        state = get_input(encoded_board, is_white_turn)
        print(unique_states)
        if state in unique_states:
            probs_list = unique_states[state]
        else:
            model = KNN(unique_states, state, int(is_white_turn))
            probs_list = model.get_average_topk(5)
        src_cells, dst_cells = parse_cells(probs_list, is_white_turn, BOARD)
        sorted_legal_moves = get_sorted_legal_moves(src_cells, dst_cells, BOARD)
        print(sorted_legal_moves)
        chosen_move = choose_move(sorted_legal_moves)
        make_move(chosen_move, BOARD)
        print(f"MODEL-{translator[move_count%2]} moved from {chosen_move[0:2]} to {chosen_move[2:]}")
    else:
        valid_input = False
        while not valid_input:
            input_move = input("Please enter a move\n")
            input_move = input_move.lower()
            is_valid_move = check_valid_move(input_move, BOARD)
            if is_valid_move:
                make_move(input_move, BOARD)
                valid_input = True
                print(f"Player moved from {input_move[0:2]} to {input_move[2:]}")
    move_count += 1

outcome = BOARD.outcome()  # Retrieves the game outcome
if outcome.winner is None:
    print("The game is a draw.")
elif outcome.winner:
    print("White wins!")
else:
    print("Black wins!")


