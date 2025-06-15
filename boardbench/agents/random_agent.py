import random
from typing import Any, List

import numpy as np

from boardbench.agents.base import Agent
from boardbench.games.base import Game


class RandomAgent(Agent):
    """
    Agent that selects moves randomly from available legal moves.
    
    Useful as a baseline or for testing the game mechanics.
    """
    
    def __init__(self, name: str = "RandomAgent", seed: int = None):
        """
        Initialize a random agent.
        
        Args:
            name: Name of the agent
            seed: Random seed for reproducibility
        """
        self._name = name
        
        if seed is not None:
            random.seed(seed)
    
    @property
    def name(self) -> str:
        return self._name
    
    def make_move(self, game: Game, state: np.ndarray, legal_moves: List[Any], player: int) -> Any:
        """
        Choose a random move from the list of legal moves.
        
        Args:
            game: The game being played
            state: The current state of the game
            legal_moves: List of legal moves available
            player: The player number this agent is playing as
            
        Returns:
            A randomly selected move
        """
        if not legal_moves:
            raise ValueError("No legal moves available")
            
        return random.choice(legal_moves)
