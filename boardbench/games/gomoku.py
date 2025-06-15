import numpy as np
from typing import List, Optional, Dict, Any, Tuple

from boardbench.games.base import Game


class Gomoku(Game):
    """
    Implementation of the Gomoku (Five in a Row) game.
    
    Players take turns placing stones on a grid.
    The objective is to be the first to form an unbroken line of five stones horizontally,
    vertically, or diagonally.
    """
    
    def __init__(self, board_size: int = 15, win_length: int = 5):
        """
        Initialize a Gomoku game.
        
        Args:
            board_size: Size of the (square) board
            win_length: Number of connected stones needed to win
        """
        self._board_size = board_size
        self._win_length = win_length
        self._current_player = 0
    
    @property
    def name(self) -> str:
        return "Gomoku"
    
    @property
    def num_players(self) -> int:
        return 2
    
    def reset(self) -> np.ndarray:
        """Reset the game to an empty board."""
        # 0 = empty, 1 = player 1, 2 = player 2
        state = np.zeros((self._board_size, self._board_size), dtype=np.int8)
        self._current_player = 0
        return state
    
    def get_legal_moves(self, state: np.ndarray, player: int) -> List[Tuple[int, int]]:
        """
        Get a list of legal moves (empty positions on the board).
        
        Args:
            state: The current game state
            player: The player whose turn it is
            
        Returns:
            List of (row, column) tuples representing empty positions
        """
        return [(r, c) for r in range(self._board_size) 
                for c in range(self._board_size) if state[r, c] == 0]
    
    def make_move(self, state: np.ndarray, move: Tuple[int, int], player: int) -> np.ndarray:
        """
        Place a stone at the specified position.
        
        Args:
            state: The current game state
            move: The (row, column) position to place the stone
            player: The player making the move
            
        Returns:
            The new state after the move
        """
        new_state = state.copy()
        row, col = move
        
        # Validate move
        if not (0 <= row < self._board_size and 0 <= col < self._board_size):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        
        if new_state[row, col] != 0:
            raise ValueError(f"Position ({row}, {col}) is already occupied")
        
        # Place the stone (player number + 1)
        new_state[row, col] = player + 1
        
        return new_state
    
    def is_terminal(self, state: np.ndarray) -> bool:
        """
        Check if the game is over.
        
        Args:
            state: The game state to check
            
        Returns:
            True if the game is over, False otherwise
        """
        # Check if there's a winner
        if self.get_winner(state) is not None:
            return True
        
        # Check if the board is full
        return len(self.get_legal_moves(state, 0)) == 0
    
    def _check_line(self, state: np.ndarray, start_row: int, start_col: int, 
                   row_dir: int, col_dir: int, player_id: int) -> bool:
        """
        Check if there is a winning line starting from (start_row, start_col) in the direction
        specified by (row_dir, col_dir).
        """
        count = 0
        row, col = start_row, start_col
        
        while (0 <= row < self._board_size and 
               0 <= col < self._board_size and 
               state[row, col] == player_id):
            count += 1
            row += row_dir
            col += col_dir
            
            if count >= self._win_length:
                return True
        
        return False
    
    def get_winner(self, state: np.ndarray) -> Optional[int]:
        """
        Check if there's a winner.
        
        Args:
            state: The game state
            
        Returns:
            The player number (0-indexed) who won, or None if no winner
        """
        # Check for wins in all directions
        for player_id in range(1, 3):  # 1 = player 0, 2 = player 1
            # Check rows
            for row in range(self._board_size):
                for col in range(self._board_size - self._win_length + 1):
                    if np.all(state[row, col:col+self._win_length] == player_id):
                        return player_id - 1
            
            # Check columns
            for col in range(self._board_size):
                for row in range(self._board_size - self._win_length + 1):
                    if np.all(state[row:row+self._win_length, col] == player_id):
                        return player_id - 1
            
            # Check diagonals (top-left to bottom-right)
            for row in range(self._board_size - self._win_length + 1):
                for col in range(self._board_size - self._win_length + 1):
                    if np.all([state[row+i, col+i] == player_id for i in range(self._win_length)]):
                        return player_id - 1
            
            # Check diagonals (bottom-left to top-right)
            for row in range(self._win_length - 1, self._board_size):
                for col in range(self._board_size - self._win_length + 1):
                    if np.all([state[row-i, col+i] == player_id for i in range(self._win_length)]):
                        return player_id - 1
        
        return None  # No winner
    
    def get_state_representation(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Convert the internal state to a human-readable representation.
        
        Args:
            state: The game state
            
        Returns:
            A dictionary representation of the state
        """
        return {
            "board": state.tolist(),
            "board_size": self._board_size,
            "win_length": self._win_length,
        }
    
    def display_state(self, state: np.ndarray) -> str:
        """
        Get a string representation of the board.
        
        Args:
            state: The game state
            
        Returns:
            A string representation of the board
        """
        # Unicode characters for the board
        EMPTY = "·"  # Middle dot
        BLACK = "○"  # White circle
        WHITE = "●"  # Black circle
        
        display = "  "
        # Column headers (numbers)
        for col in range(self._board_size):
            display += f"{col:2d}"
        display += "\n"
        
        # Board with row numbers
        for row in range(self._board_size):
            display += f"{row:2d}"
            for col in range(self._board_size):
                if state[row, col] == 0:
                    display += f" {EMPTY}"
                elif state[row, col] == 1:
                    display += f" {BLACK}"
                else:
                    display += f" {WHITE}"
            display += "\n"
        
        return display
    
    def move_to_string(self, move: Tuple[int, int]) -> str:
        """
        Convert a move to a human-readable string representation.
        
        Args:
            move: The move (row, col)
            
        Returns:
            String representation of the move
        """
        row, col = move
        return f"{row},{col}"
    
    def string_to_move(self, move_str: str) -> Tuple[int, int]:
        """
        Convert a string representation of a move back to the move format.
        
        Args:
            move_str: The string representation of the move
            
        Returns:
            The move as (row, col)
        """
        try:
            parts = move_str.split(",")
            if len(parts) != 2:
                raise ValueError(f"Invalid move format: {move_str}. Expected 'row,col'")
            
            row, col = int(parts[0]), int(parts[1])
            return (row, col)
        except ValueError:
            raise ValueError(f"Invalid move format: {move_str}. Expected integers 'row,col'")
