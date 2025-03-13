import os
import re
import uuid
import json

DIRECTORY = "./games"


def _extract_data(content, data):
    arr = content.split("[Event")[1:]
    for item in arr:
        item = "[Event" + item
        moves, score = extract_san_and_score(item)
        if moves is not None and score is not None:
            white_score, black_score = score.split("-")
            count_white = white_score in {"1", "1/2"}
            count_black = black_score in {"1", "1/2"}
            data[str(uuid.uuid4())] = {'count_white': count_white, 'count_black': count_black, 'moves': moves}


def extract_san_and_score(pgn_text):
    """
    Extracts all move SAN from a full PGN string that includes header tags and comments,
    and separately extracts the game result (score).

    This function removes:
      - PGN header lines (lines starting with '[')
      - All comments (text enclosed in curly braces, including clock annotations)
      - Move numbers (e.g., "1.", "2.", etc.)

    Parameters:
        pgn_text (str): A string containing the full PGN data.

    Returns:
        tuple: A tuple (moves, score) where moves is a list of SAN moves as strings,
               and score is a string representing the game result (e.g., "1-0", "0-1", "1/2-1/2"),
               or None if not found.
    """
    moves_text = re.sub(r'^\[.*\]\s*', '', pgn_text, flags=re.MULTILINE).strip()
    moves_text = re.sub(r'\{[^}]*\}', '', moves_text)
    score_match = re.search(r'(1-0|0-1|1/2-1/2)', moves_text)
    score = score_match.group(0) if score_match else None
    text_no_numbers = re.sub(r'\d+\.', '', moves_text)
    tokens = text_no_numbers.split()
    moves = [token for token in tokens if token not in {"1-0", "0-1", "1/2-1/2"}]
    return moves, score


def convert_pgn_to_data():
    for player in os.listdir(DIRECTORY):
        player_path = os.path.join(DIRECTORY, player)
        data = {'number_of_games': 0, 'games': {}}
        for game in os.listdir(player_path):
            game_path = os.path.join(player_path, game)
            content = read_file_to_string(game_path)
            _extract_data(content, data['games'])
        data['number_of_games'] = len(data['games'])
        file_path = f"./data/{player}.json"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)


def read_file_to_string(file_path):
    """
    Reads the entire contents of a file into a string.

    Parameters:
        file_path (str): The path to the file.

    Returns:
        str: The entire content of the file as a string.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    return content


convert_pgn_to_data()