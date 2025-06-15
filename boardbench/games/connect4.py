import numpy as np
from typing import List, Optional, Dict, Any, Tuple

from boardbench.games.base import Game


class Connect4(Game):
    """
    Implementation of the Connect4 game.
    
    Players take turns dropping colored discs into a 7-column, 6-row grid.
    The objective is to be the first to form a horizontal, vertical, or diagonal line of four discs.
    """
    
    def __init__(self, rows: int = 6, cols: int = 7, win_length: int = 4):
        """
        Initialize a Connect4 game.
        
        Args:
            rows: Number of rows on the board
            cols: Number of columns on the board
            win_length: Number of connected pieces needed to win
        """
        self._rows = rows
        self._cols = cols
        self._win_length = win_length
        self._current_player = 0
    
    @property
    def name(self) -> str:
        return "Connect4"
    
    @property
    def num_players(self) -> int:
        return 2
    
    def reset(self) -> np.ndarray:
        """Reset the game to an empty board."""
        # 0 = empty, 1 = player 1, 2 = player 2
        state = np.zeros((self._rows, self._cols), dtype=np.int8)
        self._current_player = 0
        return state
    
    def get_legal_moves(self, state: np.ndarray, player: int) -> List[int]:
        """
        Get a list of legal moves (column indices where a piece can be dropped).
        
        Args:
            state: The current game state
            player: The player whose turn it is
            
        Returns:
            List of column indices where a piece can be dropped
        """
        return [col for col in range(self._cols) if state[0, col] == 0]
    
    def make_move(self, state: np.ndarray, move: int, player: int) -> np.ndarray:
        """
        Drop a piece in the specified column.
        
        Args:
            state: The current game state
            move: The column index to drop the piece in
            player: The player making the move
            
        Returns:
            The new state after the move
        """
        new_state = state.copy()
        
        # Find the lowest empty row in the specified column
        for row in range(self._rows - 1, -1, -1):
            if new_state[row, move] == 0:
                # Place the piece (player number + 1)
                new_state[row, move] = player + 1
                break
        
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
    
    def get_winner(self, state: np.ndarray) -> Optional[int]:
        """
        Check if there's a winner.
        
        Args:
            state: The game state
            
        Returns:
            The player number (0-indexed) who won, or None if no winner
        """
        # Check horizontal, vertical, and diagonal lines
        for player_id in range(1, 3):  # 1 = player 0, 2 = player 1
            # Horizontal
            for row in range(self._rows):
                for col in range(self._cols - self._win_length + 1):
                    if np.all(state[row, col:col+self._win_length] == player_id):
                        return player_id - 1  # Convert to 0-indexed player
            
            # Vertical
            for row in range(self._rows - self._win_length + 1):
                for col in range(self._cols):
                    if np.all(state[row:row+self._win_length, col] == player_id):
                        return player_id - 1
            
            # Diagonal (down-right)
            for row in range(self._rows - self._win_length + 1):
                for col in range(self._cols - self._win_length + 1):
                    if np.all([state[row+i, col+i] == player_id for i in range(self._win_length)]):
                        return player_id - 1
            
            # Diagonal (up-right)
            for row in range(self._win_length - 1, self._rows):
                for col in range(self._cols - self._win_length + 1):
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
            "dimensions": {
                "rows": self._rows,
                "cols": self._cols
            },
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
        display = ""
        
        # Column numbers
        display += " "
        for col in range(self._cols):
            display += f" {col} "
        display += "\n"
        
        # Board
        for row in range(self._rows):
            display += "|"
            for col in range(self._cols):
                if state[row, col] == 0:
                    display += "   "
                elif state[row, col] == 1:
                    display += " X "
                else:
                    display += " O "
            display += "|\n"
        
        # Bottom
        display += "+"
        for col in range(self._cols):
            display += "---"
        display += "+\n"
        
        return display
    
    def move_to_string(self, move: int) -> str:
        """
        Convert a move to a human-readable string representation.
        
        Args:
            move: The move (column index)
            
        Returns:
            String representation of the move
        """
        return f"Column {move}"
    
    def string_to_move(self, move_str: str) -> int:
        """
        Convert a string representation of a move back to the move format.
        
        Args:
            move_str: The string representation of the move
            
        Returns:
            The move (column index)
        """
        # Parse "Column X" format
        try:
            if move_str.lower().startswith("column "):
                return int(move_str[7:])
            else:
                return int(move_str)
        except ValueError:
            raise ValueError(f"Invalid move format: {move_str}")
