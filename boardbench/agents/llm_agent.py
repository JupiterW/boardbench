import time
from typing import Any, List, Dict, Optional

import numpy as np
import json

from boardbench.agents.base import Agent
from boardbench.games.base import Game

# Import optional OpenAI dependency - handle gracefully if not installed
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMAgent(Agent):
    """
    Agent that uses a Large Language Model to make decisions.
    
    This agent formulates prompts with game state and rules,
    then uses an LLM to select moves.
    """
    
    def __init__(
        self, 
        name: str = "LLMAgent",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize an LLM agent.
        
        Args:
            name: Name of the agent
            model: LLM model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (if None, will try to load from environment)
            temperature: Sampling temperature for the model
            max_retries: Maximum number of retry attempts for API calls
            retry_delay: Delay between retry attempts (in seconds)
            system_prompt: Custom system prompt to use instead of the default
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is required for LLMAgent. Install with: pip install openai")
        
        self._name = name
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        
        # Initialize OpenAI client
        self._client = openai.OpenAI(api_key=api_key)
        
        # Set up system prompt
        if system_prompt is None:
            self._system_prompt = (
                "You are an expert game-playing AI. "
                "You will be given the current state of a board game and a list of legal moves. "
                "Your task is to analyze the game state and choose the best move from the available options. "
                "Explain your reasoning briefly and then clearly state your chosen move in the required format."
            )
        else:
            self._system_prompt = system_prompt
    
    @property
    def name(self) -> str:
        return self._name
    
    def make_move(self, game: Game, state: np.ndarray, legal_moves: List[Any], player: int) -> Any:
        """
        Use the LLM to choose a move.
        
        Args:
            game: The game being played
            state: The current state of the game
            legal_moves: List of legal moves available
            player: The player number this agent is playing as
            
        Returns:
            The selected move
        """
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Construct the prompt for the LLM
        prompt = self._create_prompt(game, state, legal_moves, player)
        
        # Query the LLM
        response = self._query_llm(prompt)
        
        # Parse the response to extract the chosen move
        chosen_move = self._parse_response(response, game, legal_moves)
        
        return chosen_move
    
    def _create_prompt(self, game: Game, state: np.ndarray, legal_moves: List[Any], player: int) -> str:
        """Create a prompt describing the game state and legal moves."""
        # Display the current game state as a string
        board_display = game.display_state(state)
        
        # Format the legal moves for display
        moves_display = "\n".join([
            f"- Move {i}: {game.move_to_string(move)}"
            for i, move in enumerate(legal_moves)
        ])
        
        # Create the user prompt
        prompt = f"""
Game: {game.name}
Player: {player} ({'X' if player == 0 else 'O'})

Current Board:
{board_display}

Legal Moves (pick one by index):
{moves_display}

Analyze the board position and choose the best move. Provide a brief explanation of your reasoning.
Then, clearly indicate your chosen move by writing "MOVE: " followed by the index of the chosen move.
For example: "MOVE: 2"
"""
        return prompt
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with the given prompt."""
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self._temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < self._max_retries - 1:
                    print(f"API call failed: {str(e)}. Retrying in {self._retry_delay} seconds...")
                    time.sleep(self._retry_delay)
                else:
                    raise e
    
    def _parse_response(self, response: str, game: Game, legal_moves: List[Any]) -> Any:
        """Parse the LLM response to extract the chosen move."""
        # Look for "MOVE: X" pattern in the response
        import re
        
        match = re.search(r"MOVE:\s*(\d+)", response, re.IGNORECASE)
        if match:
            move_index = int(match.group(1))
            
            if 0 <= move_index < len(legal_moves):
                return legal_moves[move_index]
        
        # If we didn't find a valid move, fall back to a random choice
        import random
        print("Failed to parse valid move from LLM response. Using fallback random choice.")
        return random.choice(legal_moves)
