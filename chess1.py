import chess
import chess.engine
import numpy as np
import random
import time

# Initialize the chess engine (replace with your engine path)
try:
    engine = chess.engine.SimpleEngine.popen_uci(r"D:\Ai profile\Project 10 Reinforcement Learning Agent\stockfish-windows-x86-64\stockfish\stockfish-windows-x86-64.exe") # Replace with your engine path
except FileNotFoundError:
    print("Stockfish engine not found. Please download and provide the correct path.")
    exit()

# Discretization (simplified)
def discretize_state(board):
    piece_count = {piece_type: len(board.pieces(piece_type, color)) for piece_type in range(1, 7) for color in [chess.WHITE, chess.BLACK]}
    return tuple(piece_count.values())

# Q-table (dictionary)
q_table = {}

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
epsilon_decay_rate = 0.0001
min_epsilon = 0.01
num_episodes = 5000

def get_q_value(state, action):
    return q_table.get((state, action), 0)

def update_q_value(state, action, reward, next_state, board):
    legal_next_moves = list(board.legal_moves)
    if legal_next_moves:
        best_next_q = max(get_q_value(next_state, next_action) for next_action in legal_next_moves)
    else:
        best_next_q = 0 # Handle terminal state
    q_table[(state, action)] = get_q_value(state, action) + learning_rate * (reward + discount_factor * best_next_q - get_q_value(state, action))

for episode in range(num_episodes):
    board = chess.Board()
    state = discretize_state(board)
    total_reward = 0
    while not board.is_game_over():
        legal_moves = list(board.legal_moves)
        if not legal_moves:  # Handle cases where there are no legal moves
            break

        if random.uniform(0, 1) < epsilon:
            action = random.choice(legal_moves)
        else:
            q_values = {move: get_q_value(state, move) for move in legal_moves}
            action = max(q_values, key=q_values.get)

        board.push(action)
        next_state = discretize_state(board)

        if board.is_checkmate():
            reward = 100
        elif board.is_stalemate():
            reward = -50
        else:
            reward = 0

        update_q_value(state, action, reward, next_state, board)
        state = next_state
        total_reward += reward

        # Visualization (print board)
        print(board)
        time.sleep(0.1)

    epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)
    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

engine.quit()
print("Training finished.")

# Playing the trained agent (against a random player for simplicity)
board = chess.Board()
while not board.is_game_over():
    if board.turn == chess.WHITE: # Agent's turn
      legal_moves = list(board.legal_moves)
      if legal_moves:
        state = discretize_state(board)
        q_values = {move: get_q_value(state, move) for move in legal_moves}
        action = max(q_values, key=q_values.get)
      else:
        break
    else: # Random opponent's turn
        legal_moves = list(board.legal_moves)
        if legal_moves:
            action = random.choice(legal_moves)
        else:
            break
    board.push(action)
    print(board)
    time.sleep(0.5)

if board.is_checkmate():
    print("Checkmate!")
elif board.is_stalemate():
    print("Stalemate!")
else:
    print("Game Drawn or ended by other means")