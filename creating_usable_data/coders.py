import chess


def decode_move(move_san, board=None):
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