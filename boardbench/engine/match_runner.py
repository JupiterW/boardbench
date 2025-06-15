from typing import List, Dict, Any, Optional, Tuple
import time
import json
import os
import uuid
from datetime import datetime

import numpy as np

from boardbench.games.base import Game
from boardbench.agents.base import Agent
from boardbench.utils.logger import Logger


class MatchRunner:
    """
    Core engine for running matches between agents.
    
    This class handles the game flow, enforces rules, and records match data.
    """
    
    def __init__(self, game: Game, agents: List[Agent], logger: Optional[Logger] = None):
        """
        Initialize a match runner.
        
        Args:
            game: The game to be played
            agents: List of agents participating in the game (order matters)
            logger: Optional logger for match data
        """
        self.game = game
        self.agents = agents
        
        if len(agents) != game.num_players:
            raise ValueError(f"{game.name} requires {game.num_players} players, but {len(agents)} agents were provided")
        
        # Create a default logger if none provided
        self.logger = logger or Logger(os.path.join("boardbench", "logs"))
        
        # Match metadata
        self.match_id = str(uuid.uuid4())
        self.moves_history = []
        self.states_history = []
        self.move_times = []
    
    def run_match(self, max_moves: int = 1000, verbose: bool = True) -> Dict[str, Any]:
        """
        Run a full match between the provided agents.
        
        Args:
            max_moves: Maximum number of moves before declaring a draw
            verbose: If True, display game state after each move
            
        Returns:
            Dict containing match results and statistics
        """
        # Initialize the game
        state = self.game.reset()
        self.states_history.append(np.copy(state))
        
        # Notify agents of game start
        for i, agent in enumerate(self.agents):
            agent.game_start(self.game, i)
        
        current_player = 0
        move_count = 0
        winner = None
        
        # Main game loop
        while not self.game.is_terminal(state) and move_count < max_moves:
            # Get current agent
            agent = self.agents[current_player]
            
            # Get legal moves
            legal_moves = self.game.get_legal_moves(state, current_player)
            
            if not legal_moves:
                # No legal moves available, skip turn
                if verbose:
                    print(f"Player {current_player} ({agent.name}) has no legal moves and must pass.")
                current_player = (current_player + 1) % self.game.num_players
                continue
            
            # Ask agent for move
            start_time = time.time()
            move = None
            move_success = False
            error_message = ""
            
            try:
                move = agent.make_move(self.game, state, legal_moves, current_player)
                
                if move in legal_moves:
                    move_success = True
                else:
                    error_message = f"Move {self.game.move_to_string(move)} is not in the list of legal moves"
            except Exception as e:
                error_message = f"Agent error: {str(e)}"
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Provide feedback to the agent
            agent.move_feedback(self.game, state, move, move_success, error_message)
            
            # If move is valid, apply it
            if move_success:
                # Record move
                move_info = {
                    "player": current_player,
                    "agent_name": agent.name,
                    "move": self.game.move_to_string(move),
                    "time_taken": elapsed
                }
                self.moves_history.append(move_info)
                self.move_times.append(elapsed)
                
                # Apply move
                new_state = self.game.make_move(state, move, current_player)
                state = new_state
                self.states_history.append(np.copy(state))
                
                # Display if verbose
                if verbose:
                    move_str = self.game.move_to_string(move)
                    print(f"Player {current_player} ({agent.name}) plays: {move_str} (in {elapsed:.3f}s)")
                    print(self.game.display_state(state))
                    print("=" * 40)
                
                # Switch to next player
                current_player = (current_player + 1) % self.game.num_players
                move_count += 1
            else:
                # Invalid move
                if verbose:
                    print(f"Invalid move from Player {current_player} ({agent.name}): {error_message}")
                
                # Let the agent try again or apply a penalty based on your preferred rules
                # For now, we'll give them a random legal move
                if legal_moves:
                    import random
                    move = random.choice(legal_moves)
                    
                    if verbose:
                        print(f"Choosing random legal move instead: {self.game.move_to_string(move)}")
                    
                    new_state = self.game.make_move(state, move, current_player)
                    state = new_state
                    self.states_history.append(np.copy(state))
                
                # Switch to next player
                current_player = (current_player + 1) % self.game.num_players
                move_count += 1
        
        # Game over
        winner = self.game.get_winner(state)
        
        # Notify agents of game end
        for i, agent in enumerate(self.agents):
            agent.game_end(self.game, state, winner, i)
        
        # Prepare results
        result = {
            "match_id": self.match_id,
            "game": self.game.name,
            "date": datetime.now().isoformat(),
            "agents": [agent.name for agent in self.agents],
            "moves": len(self.moves_history),
            "winner": winner,
            "avg_move_time": sum(self.move_times) / len(self.move_times) if self.move_times else 0,
            "max_move_time": max(self.move_times) if self.move_times else 0,
        }
        
        # Log the match
        self.log_match(result)
        
        if verbose:
            if winner is not None:
                winner_name = self.agents[winner].name
                print(f"Game over! Player {winner} ({winner_name}) wins in {move_count} moves!")
            else:
                print(f"Game over! It's a draw after {move_count} moves!")
        
        return result
    
    def log_match(self, result: Dict[str, Any]) -> str:
        """
        Log match results and history to a file.
        
        Args:
            result: The match result dictionary
            
        Returns:
            The path to the log file
        """
        # Create a complete log including moves and states
        log_data = {
            **result,
            "moves_history": self.moves_history,
            "final_state": self.game.get_state_representation(self.states_history[-1]),
        }
        
        # Save to log file via the logger
        log_file = self.logger.log_match(log_data)
        
        return log_file
