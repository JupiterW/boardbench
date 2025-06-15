"""
Unit tests for the RandomAgent implementation.
"""

import pytest
import numpy as np
import random
from unittest.mock import MagicMock

from boardbench.agents.random_agent import RandomAgent
from boardbench.games.gomoku import Gomoku


class TestRandomAgent:
    """Test the functionality of the RandomAgent class."""
    
    def test_initialization(self):
        """Test that the agent initializes correctly."""
        # Test with default parameters
        agent = RandomAgent()
        assert agent.name == "RandomAgent"
        
        # Test with custom name
        custom_agent = RandomAgent(name="CustomRandomBot")
        assert custom_agent.name == "CustomRandomBot"
        
        # Test with seed
        seeded_agent = RandomAgent(seed=42)
        assert seeded_agent.name == "RandomAgent"
    
    def test_make_move_returns_legal_move(self, gomoku_game, empty_board_15x15):
        """Test that make_move returns a move from the legal moves list."""
        agent = RandomAgent(seed=42)
        legal_moves = [(0, 0), (1, 1), (2, 2)]
        
        move = agent.make_move(gomoku_game, empty_board_15x15, legal_moves, 0)
        
        assert move in legal_moves
    
    def test_make_move_with_fixed_seed(self):
        """Test that agents with the same seed make the same random choice."""
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=42)
        
        # Create a mock game and state
        mock_game = MagicMock()
        mock_state = np.zeros((5, 5))
        
        # Define the same legal moves for both
        legal_moves = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        
        # Both agents should make the same choice with the same seed
        move1 = agent1.make_move(mock_game, mock_state, legal_moves, 0)
        
        # Reset the random seed for agent2 (since it was set in __init__)
        random.seed(42)
        move2 = agent2.make_move(mock_game, mock_state, legal_moves, 0)
        
        assert move1 == move2
    
    def test_make_move_no_legal_moves(self, gomoku_game, empty_board_15x15):
        """Test that make_move raises an error when no legal moves are available."""
        agent = RandomAgent()
        
        with pytest.raises(ValueError) as excinfo:
            agent.make_move(gomoku_game, empty_board_15x15, [], 0)
        
        assert "No legal moves available" in str(excinfo.value)
    
    def test_make_move_distribution(self, gomoku_game, empty_board_15x15):
        """Test that over multiple calls, make_move uses different moves from the legal list."""
        agent = RandomAgent(seed=42)
        legal_moves = [(i, i) for i in range(5)]  # 5 different legal moves
        
        # Make many moves and count frequencies
        move_counts = {move: 0 for move in legal_moves}
        trials = 100
        
        for _ in range(trials):
            move = agent.make_move(gomoku_game, empty_board_15x15, legal_moves, 0)
            move_counts[move] += 1
        
        # Given enough trials, all moves should be chosen at least once
        for move, count in move_counts.items():
            assert count > 0, f"Move {move} was never selected in {trials} trials"
    
    def test_optional_methods_exist(self):
        """Test that the optional agent methods exist and don't raise errors."""
        agent = RandomAgent()
        mock_game = MagicMock()
        mock_state = np.zeros((5, 5))
        
        # These methods should exist but don't need to do anything
        agent.game_start(mock_game, 0)
        agent.game_end(mock_game, mock_state, 0, 0)
        agent.move_feedback(mock_game, mock_state, (0, 0), True, "")
