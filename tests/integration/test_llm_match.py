"""
Integration tests for matches involving the LLM agent.
"""

import pytest
from unittest.mock import patch, Mock
import numpy as np
import json

from boardbench.engine.match_runner import MatchRunner
from boardbench.games.gomoku import Gomoku
from boardbench.agents.llm_agent import LLMAgent, OPENAI_AVAILABLE
from boardbench.agents.enforced_random_agent import EnforcedRandomAgent


# Skip all tests in this module if OpenAI package is not installed
pytestmark = pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI package is not installed")


class TestLLMMatchIntegration:
    """Test matches involving the LLM agent."""

    @pytest.fixture
    def mock_openai_completion(self):
        """Create a mock for OpenAI completions that returns valid move indices."""
        with patch('openai.OpenAI') as mock_client:
            # Configure the mock to return responses that include valid move indices
            mock_instance = mock_client.return_value
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message = Mock()
            
            # Will be customized in each test
            mock_completion.choices[0].message.content = "MOVE: 0"
            mock_instance.chat.completions.create.return_value = mock_completion
            
            yield mock_instance

    @pytest.fixture
    def small_game(self):
        """Create a small 5x5 Gomoku game for faster testing."""
        return Gomoku(board_size=5, win_length=3)

    @pytest.fixture
    def patched_logger(self):
        """Mock the logger to avoid file operations during tests."""
        # Simply patch the Logger.log_match method which is called when logging match results
        with patch('boardbench.utils.logger.Logger.log_match') as mock_log_match:
            mock_log_match.return_value = "mocked_log_path"
            yield mock_log_match

    def test_llm_vs_enforced_random(self, mock_openai_completion, small_game, patched_logger):
        """Test a match between LLM agent and EnforcedRandomAgent."""
        # Create the agents
        llm_agent = LLMAgent(name="TestLLM", api_key="fake-key")
        random_agent = EnforcedRandomAgent(name="TestEnforcedRandom", seed=42)
        
        # Configure the mock to return different responses for each move
        # This will make the LLM agent choose the first legal move each time
        mock_openai_completion.chat.completions.create.return_value.choices[0].message.content = (
            "I'll choose the first available move.\n\nMOVE: 0"
        )
        
        # Set up the match runner with mocked logger
        match_runner = MatchRunner(small_game, [llm_agent, random_agent])
        
        # Run a short match with max 6 moves
        result = match_runner.run_match(max_moves=6, verbose=False)
        
        # Verify the match completed successfully
        assert result is not None
        assert "moves" in result
        assert "match_id" in result
        assert "agents" in result
        assert result["game"] == "Gomoku"
        assert result["agents"][0] == "TestLLM"
        assert result["agents"][1] == "TestEnforcedRandom"
        
        # Verify the LLM API was called for LLM agent's turns
        expected_min_calls = result["moves"] // 2  # LLM agent plays every other turn
        assert mock_openai_completion.chat.completions.create.call_count >= expected_min_calls
        
        # Verify logging occurred
        assert patched_logger.call_count == 1

    def test_llm_agent_move_validation(self, mock_openai_completion, small_game):
        """Test that the LLM agent's moves are properly validated."""
        # Create the agents
        llm_agent = LLMAgent(name="TestLLM", api_key="fake-key")
        
        # Mock a 5x5 game board with some existing moves
        # Player 0 (X): (0,0), (1,1)
        # Player 1 (O): (0,1), (1,0)
        board = np.zeros((5, 5), dtype=np.int8)
        board[0, 0] = 1  # Player 0's move
        board[0, 1] = 2  # Player 1's move
        board[1, 0] = 2  # Player 1's move
        board[1, 1] = 1  # Player 0's move
        
        # Available legal moves - (2,2) should be a valid open position
        player = 0  # Player 0's turn
        legal_moves = small_game.get_legal_moves(board, player)
        assert (2, 2) in legal_moves
        
        # Configure the mock for two different scenarios
        # Scenario 1: LLM returns a valid move index within legal_moves
        mock_openai_completion.chat.completions.create.return_value.choices[0].message.content = (
            "I choose position (2,2).\n\nMOVE: 5"  # Assuming (2,2) is at index 5
        )
        
        # Ensure (2,2) is at the expected index for testing
        # Find the index of (2,2) in legal_moves
        try:
            target_index = legal_moves.index((2, 2))
            # Update the mocked response to use the correct index
            mock_openai_completion.chat.completions.create.return_value.choices[0].message.content = (
                f"I choose position (2,2).\n\nMOVE: {target_index}"
            )
        except ValueError:
            # If (2,2) is not in legal_moves, just use index 0
            pass
        
        # Test the agent's move selection
        move = llm_agent.make_move(small_game, board, legal_moves, player)
        
        # Should get a valid move from legal_moves
        assert move in legal_moves
        
        # API should have been called
        assert mock_openai_completion.chat.completions.create.call_count > 0
