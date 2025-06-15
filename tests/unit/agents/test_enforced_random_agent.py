import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from boardbench.agents.enforced_random_agent import EnforcedRandomAgent


class TestEnforcedRandomAgent(unittest.TestCase):
    """Test cases for the EnforcedRandomAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = EnforcedRandomAgent(seed=42)
        self.game = MagicMock()
        self.state = np.zeros((3, 3), dtype=np.int8)  # Simple 3x3 board
        self.player = 1

    def test_init(self):
        """Test initialization parameters."""
        agent1 = EnforcedRandomAgent(name="TestAgent", seed=123, max_retries=5)
        self.assertEqual(agent1.name, "TestAgent")
        self.assertEqual(agent1.max_retries, 5)

    def test_make_move_with_valid_moves(self):
        """Test selecting from valid moves."""
        legal_moves = [(0, 0), (0, 1), (1, 1)]
        
        # All moves are considered valid in test state
        self.game.make_move.return_value = self.state
        
        move = self.agent.make_move(self.game, self.state, legal_moves, self.player)
        
        # The move should be one of the legal moves
        self.assertIn(move, legal_moves)
        
        # The game's make_move should have been called for validation
        self.game.make_move.assert_called()

    def test_make_move_with_empty_move_list(self):
        """Test handling empty move list."""
        with self.assertRaises(ValueError):
            self.agent.make_move(self.game, self.state, [], self.player)

    def test_make_move_filters_invalid_moves(self):
        """Test filtering out invalid moves during validation."""
        legal_moves = [(0, 0), (0, 1), (1, 1), (2, 2)]
        
        # Make (1,1) and (2,2) invalid moves that raise ValueError
        def mock_make_move(state, move, player):
            if move in [(1, 1), (2, 2)]:
                raise ValueError("Invalid move")
            return state
        
        self.game.make_move.side_effect = mock_make_move
        
        # Patch random.choice to always return the first option
        with patch('random.choice', return_value=(0, 0)):
            move = self.agent.make_move(self.game, self.state, legal_moves, self.player)
            
        # The move should be one of the valid moves
        self.assertIn(move, [(0, 0), (0, 1)])
        self.assertNotIn(move, [(1, 1), (2, 2)])

    def test_make_move_all_moves_invalid(self):
        """Test case where all moves are invalid after validation."""
        legal_moves = [(0, 0), (0, 1)]
        
        # Make all moves raise ValueError
        def mock_make_move(state, move, player):
            raise ValueError("Invalid move")
        
        self.game.make_move.side_effect = mock_make_move
        
        with self.assertRaises(ValueError):
            self.agent.make_move(self.game, self.state, legal_moves, self.player)

    def test_move_feedback(self):
        """Test move feedback handling."""
        # Prepare test data
        game = MagicMock()
        state = np.zeros((3, 3))
        move = (0, 0)
        
        # Test successful move feedback
        with patch('builtins.print') as mock_print:
            self.agent.move_feedback(game, state, move, True, "")
            mock_print.assert_not_called()
        
        # Test unsuccessful move feedback
        with patch('builtins.print') as mock_print:
            self.agent.move_feedback(game, state, move, False, "Test error")
            mock_print.assert_called()
