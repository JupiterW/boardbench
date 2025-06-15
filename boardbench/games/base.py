from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any, Dict
import numpy as np


class Game(ABC):
    """
    Base abstract class for all board games.
    
    All games must implement this interface to work with the BoardBench framework.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the game."""
        pass
    
    @property
    @abstractmethod
    def num_players(self) -> int:
        """Return the number of players required for this game."""
        pass
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the game to its initial state.
        
        Returns:
            np.ndarray: The initial state of the game.
        """
        pass
    
    @abstractmethod
    def get_legal_moves(self, state: np.ndarray, player: int) -> List[Any]:
        """
        Get a list of legal moves for the current player.
        
        Args:
            state (np.ndarray): The current game state.
            player (int): The player whose turn it is.
            
        Returns:
            List[Any]: A list of legal moves in a game-specific format.
        """
        pass
    
    @abstractmethod
    def make_move(self, state: np.ndarray, move: Any, player: int) -> np.ndarray:
        """
        Apply a move to the current state and return the new state.
        
        Args:
            state (np.ndarray): The current game state.
            move (Any): The move to apply.
            player (int): The player making the move.
            
        Returns:
            np.ndarray: The new state after the move.
        """
        pass
    
    @abstractmethod
    def is_terminal(self, state: np.ndarray) -> bool:
        """
        Check if the state is terminal (game over).
        
        Args:
            state (np.ndarray): The game state to check.
            
        Returns:
            bool: True if the game is over, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_winner(self, state: np.ndarray) -> Optional[int]:
        """
        Get the winner of the game if any.
        
        Args:
            state (np.ndarray): The game state.
            
        Returns:
            Optional[int]: The player number who won, or None if no winner or draw.
        """
        pass
    
    @abstractmethod
    def get_state_representation(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Convert the internal state to a human-readable representation.
        
        Args:
            state (np.ndarray): The game state.
            
        Returns:
            Dict[str, Any]: A dictionary representation of the state.
        """
        pass
    
    @abstractmethod
    def display_state(self, state: np.ndarray) -> str:
        """
        Get a string representation of the board for display purposes.
        
        Args:
            state (np.ndarray): The game state.
            
        Returns:
            str: A string representation of the board.
        """
        pass
    
    @abstractmethod
    def move_to_string(self, move: Any) -> str:
        """
        Convert a move to a human-readable string representation.
        
        Args:
            move (Any): The move to convert.
            
        Returns:
            str: String representation of the move.
        """
        pass
    
    @abstractmethod
    def string_to_move(self, move_str: str) -> Any:
        """
        Convert a string representation of a move back to the move format.
        
        Args:
            move_str (str): The string representation of the move.
            
        Returns:
            Any: The move in the game's internal format.
        """
        pass
