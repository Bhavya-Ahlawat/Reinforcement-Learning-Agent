# Chess Reinforcement Learning Agent

This project implements a reinforcement learning agent that plays chess using the [Stockfish](https://stockfishchess.org/) chess engine. The agent learns to play by interacting with the environment and updating its strategy based on the outcomes of its games.

## Overview

The agent uses Q-learning to improve its performance over time. It discretizes the chess board state and maintains a Q-table to store the expected rewards for each action.

## Installation

To run this project, you need to have Python installed along with the required dependencies. You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

To train the agent, run the `chess1.py` script:

```bash
python chess1.py
```

The agent will play against itself and learn from its experiences.



