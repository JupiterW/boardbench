from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

import numpy as np

from boardbench.games.base import Game


class Agent(ABC):
    """
    Base abstract class for all agents.
    
    All agents (LLMs, bots, human players) must implement this interface
    to interact with the BoardBench framework.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the agent."""
        pass
    
    @abstractmethod
    def make_move(self, game: Game, state: np.ndarray, legal_moves: List[Any], player: int) -> Any:
        """
        Decide on a move given the current game state.
        
        Args:
            game: The game being played
            state: The current state of the game
            legal_moves: List of legal moves available
            player: The player number this agent is playing as
            
        Returns:
            The chosen move in the game's move format
        """
        pass
    
    def game_start(self, game: Game, player: int) -> None:
        """
        Called when a game starts.
        
        Args:
            game: The game being played
            player: The player number this agent is playing as
        """
        pass
    
    def game_end(self, game: Game, state: np.ndarray, winner: Optional[int], player: int) -> None:
        """
        Called when a game ends.
        
        Args:
            game: The game being played
            state: The final state of the game
            winner: The player number of the winner, or None for a draw
            player: The player number this agent was playing as
        """
        pass
    
    def move_feedback(self, game: Game, state: np.ndarray, move: Any, success: bool, message: str) -> None:
        """
        Feedback about a move attempt.
        
        Args:
            game: The game being played
            state: The current state of the game
            move: The attempted move
            success: Whether the move was valid and accepted
            message: A message describing why the move failed, if applicable
        """
        pass
