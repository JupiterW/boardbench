import random
from typing import Any, List, Optional

import numpy as np

from boardbench.agents.random_agent import RandomAgent
from boardbench.games.base import Game


class EnforcedRandomAgent(RandomAgent):
    """
    Agent that selects moves randomly from available legal moves
    with additional validation to ensure only legal moves are made.
    
    Unlike the basic RandomAgent, this implementation has extra safeguards
    to prevent returning invalid moves.
    """
    
    def __init__(self, name: str = "EnforcedRandomAgent", seed: Optional[int] = None,
                 max_retries: int = 3):
        """
        Initialize an enforced random agent.
        
        Args:
            name: Name of the agent
            seed: Random seed for reproducibility
            max_retries: Maximum number of retries if a move appears invalid
        """
        super().__init__(name, seed)
        self.max_retries = max_retries
        
    def make_move(self, game: Game, state: np.ndarray, legal_moves: List[Any], player: int) -> Any:
        """
        Choose a random move from the list of legal moves with validation.
        
        Args:
            game: The game being played
            state: The current state of the game
            legal_moves: List of legal moves available
            player: The player number this agent is playing as
            
        Returns:
            A randomly selected move that is guaranteed to be legal
        """
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Ensure legal_moves is not empty
        if len(legal_moves) == 0:
            raise ValueError("Empty legal_moves list provided")
            
        # Double-check with the game that the moves are actually legal
        validated_moves = []
        for move in legal_moves:
            try:
                # Check if the move is valid by simulation (doesn't actually make the move)
                test_state = state.copy()
                game.make_move(test_state, move, player)
                validated_moves.append(move)
            except ValueError:
                # Skip any move that would cause an error
                continue
            
        if not validated_moves:
            raise ValueError("No valid moves available after validation")
            
        return random.choice(validated_moves)
    
    def move_feedback(self, game: Game, state: np.ndarray, move: Any, 
                      success: bool, message: str = "") -> None:
        """
        Process feedback about a move's validity.
        
        Args:
            game: The game being played
            state: The state when the move was made
            move: The move that was made
            success: Whether the move was successfully applied
            message: Error message if move was invalid
        """
        if not success:
            # Log the error for monitoring purposes
            print(f"[{self.name}] Received move feedback - Invalid move: {message}")
