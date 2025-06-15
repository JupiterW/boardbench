"""
Unit tests for the LLMAgent implementation.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock

from boardbench.agents.llm_agent import LLMAgent, OPENAI_AVAILABLE
from boardbench.games.gomoku import Gomoku


# Skip all tests if OpenAI is not installed
pytestmark = pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI package is not installed")


class TestLLMAgent:
    """Test the functionality of the LLM agent."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        with patch('openai.OpenAI') as mock_client:
            # Configure the mock to return a specific response when called
            mock_instance = mock_client.return_value
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message = Mock()
            mock_completion.choices[0].message.content = "I think move 2 is best because it creates a threat.\n\nMOVE: 2"
            mock_instance.chat.completions.create.return_value = mock_completion
            
            yield mock_instance
    
    def test_initialization(self, mock_openai_client):
        """Test that the LLM agent initializes correctly."""
        agent = LLMAgent(name="TestLLM", model="gpt-4", api_key="test-key")
        
        assert agent.name == "TestLLM"
        assert agent._model == "gpt-4"
        assert isinstance(agent._system_prompt, str)
        assert "expert game-playing AI" in agent._system_prompt
    
    def test_initialization_with_custom_system_prompt(self, mock_openai_client):
        """Test initialization with a custom system prompt."""
        custom_prompt = "You are a strategic game player."
        agent = LLMAgent(name="CustomPromptLLM", system_prompt=custom_prompt)
        
        assert agent._system_prompt == custom_prompt
    
    def test_make_move_valid_response(self, mock_openai_client, small_gomoku_game, empty_board_5x5):
        """Test that make_move correctly interprets a valid LLM response."""
        agent = LLMAgent(name="TestLLM")
        
        legal_moves = [(0, 0), (0, 1), (0, 2)]  # 3 legal moves
        
        # Set up the mock response to indicate move index 2
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
            "After analyzing the board, I believe move 2 is best because it creates a threat.\n\nMOVE: 2"
        )
        
        # Call make_move
        move = agent.make_move(small_gomoku_game, empty_board_5x5, legal_moves, 0)
        
        # Verify the correct move was chosen
        assert move == (0, 2)  # Index 2 in the legal_moves list
        
        # Verify the API was called with expected arguments
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4"
        assert len(call_args["messages"]) == 2
        assert call_args["messages"][0]["role"] == "system"
        assert call_args["messages"][1]["role"] == "user"
    
    def test_make_move_invalid_response(self, mock_openai_client, small_gomoku_game, empty_board_5x5):
        """Test that make_move handles invalid responses gracefully."""
        agent = LLMAgent(name="TestLLM")
        
        legal_moves = [(0, 0), (0, 1), (0, 2)]  # 3 legal moves
        
        # Set up the mock with an invalid response
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
            "I'm not sure what move to make. Let me think..."
        )
        
        # When response format is invalid, it should fall back to a random choice
        with patch('random.choice', return_value=(0, 1)):
            move = agent.make_move(small_gomoku_game, empty_board_5x5, legal_moves, 0)
            
            # Should select the fallback random move
            assert move == (0, 1)
            
            # Verify API was called
            mock_openai_client.chat.completions.create.assert_called_once()
    
    def test_make_move_out_of_range_index(self, mock_openai_client, small_gomoku_game, empty_board_5x5):
        """Test handling of out-of-range move indices."""
        agent = LLMAgent(name="TestLLM")
        
        legal_moves = [(0, 0), (0, 1)]  # Only 2 legal moves
        
        # Set up the mock to return a move index that's out of range
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
            "I'll choose move 5.\n\nMOVE: 5"
        )
        
        # When index is out of range, it should fall back to a random choice
        with patch('random.choice', return_value=(0, 0)):
            move = agent.make_move(small_gomoku_game, empty_board_5x5, legal_moves, 0)
            
            # Should select the fallback random move
            assert move == (0, 0)
    
    def test_api_retry_on_failure(self, small_gomoku_game, empty_board_5x5):
        """Test that the agent retries on API failure."""
        with patch('openai.OpenAI') as mock_client:
            # Configure the mock instance
            mock_instance = mock_client.return_value
            
            # Make the API fail twice, then succeed
            mock_instance.chat.completions.create.side_effect = [
                Exception("API error"),
                Exception("API error"),
                Mock(choices=[Mock(message=Mock(content="MOVE: 0"))])
            ]
            
            # Create agent with short retry delay
            agent = LLMAgent(name="RetryAgent", max_retries=3, retry_delay=0.001)
            
            # Call make_move
            legal_moves = [(0, 0), (0, 1)]
            move = agent.make_move(small_gomoku_game, empty_board_5x5, legal_moves, 0)
            
            # Should have retried and eventually succeeded
            assert move == (0, 0)
            assert mock_instance.chat.completions.create.call_count == 3
    
    def test_create_prompt(self, mock_openai_client, small_gomoku_game, empty_board_5x5):
        """Test prompt creation logic."""
        agent = LLMAgent(name="TestLLM")
        
        legal_moves = [(0, 0), (1, 1)]
        player = 0
        
        prompt = agent._create_prompt(small_gomoku_game, empty_board_5x5, legal_moves, player)
        
        # Verify prompt contains key elements
        assert "Game: Gomoku" in prompt
        assert "Player: 0" in prompt
        assert "Legal Moves" in prompt
        assert "Move 0:" in prompt
        assert "Move 1:" in prompt
        assert "MOVE: " in prompt  # Instructions for response format
    
    def test_parse_response(self, mock_openai_client):
        """Test parsing of different LLM responses."""
        agent = LLMAgent(name="TestLLM")
        game = Gomoku(board_size=5)
        legal_moves = [(0, 0), (1, 1), (2, 2)]
        
        # Test valid response
        valid_response = "After careful consideration, I believe the best move is 1.\nMOVE: 1"
        parsed_move = agent._parse_response(valid_response, game, legal_moves)
        assert parsed_move == (1, 1)
        
        # Test response with different formatting
        formatted_response = "I will play move:1"
        parsed_move = agent._parse_response(formatted_response, game, legal_moves)
        assert parsed_move == (1, 1)
        
        # Test invalid response (fallback to random)
        invalid_response = "I'm not sure what to do."
        with patch('random.choice', return_value=(2, 2)):
            parsed_move = agent._parse_response(invalid_response, game, legal_moves)
            assert parsed_move == (2, 2)
