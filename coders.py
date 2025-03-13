import chess


def decode_move(move_san, board=None):
    """
    Decode a chess move in standard algebraic notation (SAN) and return a tuple
    of (origin cell, destination cell). The board state is needed to disambiguate moves.
    If no board is provided, the starting position is assumed.

    Parameters:
      move_san (str): The move in algebraic notation (e.g., "e4", "Nf3", "cxd4", "O-O").
      board (chess.Board): (Optional) The board position before the move.

    Returns:
      tuple: (origin cell, destination cell) as strings, e.g., ("e2", "e4").
    """
    if board is None:
        board = chess.Board()
    try:
        move = board.parse_san(move_san)
        # Get origin and destination squares as human-readable strings
        origin = chess.square_name(move.from_square)
        destination = chess.square_name(move.to_square)
        return origin, destination
    except Exception as e:
        return f"Error parsing move '{move_san}': {e}"


def encode_move(origin, destination, board=None):
    """
    Convert an origin cell and a destination cell into a SAN move string.
    The board state is needed to disambiguate moves when necessary.
    If no board is provided, the starting position is assumed.

    Parameters:
      origin (str): The starting square (e.g., "e2").
      destination (str): The target square (e.g., "e4").
      board (chess.Board): (Optional) The board position before the move.

    Returns:
      str: The move in SAN notation (e.g., "e4", "Nf3", "O-O"), or an error message.
    """
    if board is None:
        board = chess.Board()
    try:
        from_square = chess.parse_square(origin)
        to_square = chess.parse_square(destination)
        move = chess.Move(from_square, to_square)
        if move not in board.legal_moves:
            return f"Illegal move: {origin} to {destination} in the current position."
        move_san = board.san(move)
        return move_san
    except Exception as e:
        return f"Error encoding move from '{origin}' to '{destination}': {e}"


if __name__ == "__main__":
    # Using the initial board position:
    print("Move 'e4':", decode_move("e4"))
    print("Move 'Nf3':", decode_move("Nf3"))
    print("Move 'O-O':", decode_move("O-O"))

    # Example with a custom board state:
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    # Now decode a move from this position
    print("After 1.e4 e5 2.Nf3, move 'Nc6':", decode_move("Nc6", board))