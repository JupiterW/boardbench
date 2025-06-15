"""
Pytest configuration file for BoardBench tests.

This file contains common fixtures and settings for the test suite.
"""

import os
import sys
import pytest
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project components for fixtures
from boardbench.games.base import Game
from boardbench.games.gomoku import Gomoku
from boardbench.agents.base import Agent
from boardbench.agents.random_agent import RandomAgent


@pytest.fixture
def gomoku_game():
    """Return a standard 15x15 Gomoku game instance."""
    return Gomoku(board_size=15, win_length=5)


@pytest.fixture
def small_gomoku_game():
    """Return a small 5x5 Gomoku game instance for faster testing."""
    return Gomoku(board_size=5, win_length=3)


@pytest.fixture
def random_agent():
    """Return a random agent with fixed seed for reproducibility."""
    return RandomAgent(name="TestRandomAgent", seed=42)


@pytest.fixture
def empty_board_5x5():
    """Return an empty 5x5 board."""
    return np.zeros((5, 5), dtype=np.int8)


@pytest.fixture
def empty_board_15x15():
    """Return an empty 15x15 board."""
    return np.zeros((15, 15), dtype=np.int8)


@pytest.fixture
def horizontal_win_board_5x5():
    """Return a 5x5 board with a horizontal win for player 1."""
    board = np.zeros((5, 5), dtype=np.int8)
    # Player 1 has 3 in a row (indexes 0-2) in the middle row
    board[2, 0:3] = 1
    return board


@pytest.fixture
def vertical_win_board_5x5():
    """Return a 5x5 board with a vertical win for player 2."""
    board = np.zeros((5, 5), dtype=np.int8)
    # Player 2 has 3 in a column (indexes 0-2) in the middle column
    board[0:3, 2] = 2
    return board


@pytest.fixture
def diagonal_win_board_5x5():
    """Return a 5x5 board with a diagonal win for player 1."""
    board = np.zeros((5, 5), dtype=np.int8)
    # Player 1 has 3 in a diagonal
    for i in range(3):
        board[i, i] = 1
    return board


@pytest.fixture
def full_board_5x5():
    """Return a full 5x5 board with no winner (draw)."""
    # Alternating pattern of 1s and 2s with no winning line
    board = np.zeros((5, 5), dtype=np.int8)
    for i in range(5):
        for j in range(5):
            board[i, j] = ((i + j) % 2) + 1
    return board
