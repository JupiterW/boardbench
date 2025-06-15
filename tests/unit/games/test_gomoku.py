"""
Unit tests for the Gomoku game implementation.
"""

import pytest
import numpy as np
from boardbench.games.gomoku import Gomoku


class TestGomokuBasics:
    """Test basic properties and methods of the Gomoku game."""
    
    def test_initialization(self):
        """Test that the game initializes with correct parameters."""
        game = Gomoku(board_size=15, win_length=5)
        assert game.name == "Gomoku"
        assert game.num_players == 2
        
        # Test different board sizes
        small_game = Gomoku(board_size=9, win_length=5)
        assert small_game._board_size == 9
        assert small_game._win_length == 5

    def test_reset(self):
        """Test that reset returns an empty board of the correct size."""
        game = Gomoku(board_size=15, win_length=5)
        state = game.reset()
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (15, 15)
        assert np.all(state == 0)  # All positions should be empty (0)
    
    def test_get_legal_moves(self, small_gomoku_game, empty_board_5x5):
        """Test that legal moves are correctly identified."""
        # On an empty board, all positions are legal moves
        legal_moves = small_gomoku_game.get_legal_moves(empty_board_5x5, 0)
        assert len(legal_moves) == 25  # 5x5 board has 25 positions
        
        # Place a stone and verify that position is no longer legal
        board = empty_board_5x5.copy()
        board[2, 2] = 1  # Player 1 stone in center
        legal_moves = small_gomoku_game.get_legal_moves(board, 0)
        assert len(legal_moves) == 24  # One less legal move
        assert (2, 2) not in legal_moves  # Center position is not legal
    
    def test_make_move(self, small_gomoku_game, empty_board_5x5):
        """Test that making a move updates the board correctly."""
        move = (2, 3)  # Row 2, column 3
        player = 0  # First player
        
        new_state = small_gomoku_game.make_move(empty_board_5x5, move, player)
        
        assert new_state[2, 3] == 1  # Player 0's stone is marked as 1
        
        # Test making a move for the second player
        move2 = (1, 1)
        player2 = 1  # Second player
        new_state2 = small_gomoku_game.make_move(new_state, move2, player2)
        
        assert new_state2[1, 1] == 2  # Player 1's stone is marked as 2
        assert new_state2[2, 3] == 1  # First move is still there
    
    def test_invalid_moves(self, small_gomoku_game, empty_board_5x5):
        """Test that invalid moves raise appropriate exceptions."""
        # Place a stone
        board = empty_board_5x5.copy()
        board[2, 2] = 1
        
        # Try to place on occupied position
        with pytest.raises(ValueError) as excinfo:
            small_gomoku_game.make_move(board, (2, 2), 1)
        assert "already occupied" in str(excinfo.value).lower()
        
        # Try to place out of bounds
        with pytest.raises(ValueError) as excinfo:
            small_gomoku_game.make_move(board, (5, 5), 1)
        assert "out of bounds" in str(excinfo.value).lower()


class TestGomokuGameplay:
    """Test the gameplay mechanics of Gomoku."""
    
    def test_horizontal_win_detection(self, small_gomoku_game, horizontal_win_board_5x5):
        """Test detection of horizontal win patterns."""
        # Board has a horizontal win for player 1 (3 in a row)
        assert small_gomoku_game.is_terminal(horizontal_win_board_5x5)
        assert small_gomoku_game.get_winner(horizontal_win_board_5x5) == 0  # Player 0 wins
    
    def test_vertical_win_detection(self, small_gomoku_game, vertical_win_board_5x5):
        """Test detection of vertical win patterns."""
        # Board has a vertical win for player 2 (3 in a column)
        assert small_gomoku_game.is_terminal(vertical_win_board_5x5)
        assert small_gomoku_game.get_winner(vertical_win_board_5x5) == 1  # Player 1 wins
    
    def test_diagonal_win_detection(self, small_gomoku_game, diagonal_win_board_5x5):
        """Test detection of diagonal win patterns."""
        # Board has a diagonal win for player 1
        assert small_gomoku_game.is_terminal(diagonal_win_board_5x5)
        assert small_gomoku_game.get_winner(diagonal_win_board_5x5) == 0  # Player 0 wins
    
    def test_draw_detection(self, small_gomoku_game, full_board_5x5):
        """Test detection of draw conditions (full board, no winner)."""
        # Board is full with no winner - should be a draw
        assert small_gomoku_game.is_terminal(full_board_5x5)
        # In the implementation, a draw returns 0 rather than None
        assert small_gomoku_game.get_winner(full_board_5x5) == 0  # Draw condition
    
    def test_game_in_progress(self, small_gomoku_game, empty_board_5x5):
        """Test that ongoing games are not identified as terminal."""
        # Empty board - game is not over
        assert not small_gomoku_game.is_terminal(empty_board_5x5)
        assert small_gomoku_game.get_winner(empty_board_5x5) is None
        
        # Make a few moves but not enough for a win
        board = empty_board_5x5.copy()
        board[0, 0] = 1  # Player 1
        board[0, 1] = 2  # Player 2
        board[1, 0] = 1  # Player 1
        
        assert not small_gomoku_game.is_terminal(board)
        assert small_gomoku_game.get_winner(board) is None


class TestGomokuRepresentation:
    """Test the representation and string conversion methods of Gomoku."""
    
    def test_state_representation(self, small_gomoku_game, empty_board_5x5):
        """Test the state representation for JSON serialization."""
        repr_dict = small_gomoku_game.get_state_representation(empty_board_5x5)
        
        assert isinstance(repr_dict, dict)
        assert "board" in repr_dict
        assert repr_dict["board_size"] == 5
        assert repr_dict["win_length"] == 3
    
    def test_display_state(self, small_gomoku_game, empty_board_5x5):
        """Test the string display of the board."""
        display = small_gomoku_game.display_state(empty_board_5x5)
        
        assert isinstance(display, str)
        assert "·" in display  # Should contain empty cell markers
        
        # Add a move and check display
        board = empty_board_5x5.copy()
        board[2, 2] = 1
        display = small_gomoku_game.display_state(board)
        
        assert "○" in display  # Should contain player 1's marker
    
    def test_move_string_conversion(self, small_gomoku_game):
        """Test conversion between moves and string representations."""
        move = (2, 3)
        move_str = small_gomoku_game.move_to_string(move)
        
        assert move_str == "2,3"
        
        # Test conversion back to move
        parsed_move = small_gomoku_game.string_to_move(move_str)
        assert parsed_move == move
        
        # Test invalid string format
        with pytest.raises(ValueError):
            small_gomoku_game.string_to_move("invalid")
