"""
Unit tests for the MatchRunner engine.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import os
import json

from boardbench.engine.match_runner import MatchRunner
from boardbench.games.gomoku import Gomoku
from boardbench.agents.random_agent import RandomAgent


class TestMatchRunnerInit:
    """Test the initialization of MatchRunner."""
    
    def test_initialization(self, gomoku_game, random_agent):
        """Test that the match runner initializes correctly."""
        # Create a second agent
        second_agent = RandomAgent(name="Agent2")
        agents = [random_agent, second_agent]
        
        runner = MatchRunner(gomoku_game, agents)
        
        assert runner.game == gomoku_game
        assert runner.agents == agents
        assert runner.match_id is not None
        assert runner.moves_history == []
        assert runner.states_history == []
        assert runner.move_times == []
        assert runner.logger is not None
    
    def test_initialization_with_wrong_number_of_agents(self, gomoku_game, random_agent):
        """Test error when initializing with wrong number of agents."""
        # Gomoku requires 2 players but we only provide 1
        agents = [random_agent]
        
        with pytest.raises(ValueError) as excinfo:
            MatchRunner(gomoku_game, agents)
        
        assert "requires 2 players" in str(excinfo.value)
        
        # Test with too many agents
        agents = [random_agent, RandomAgent("Agent2"), RandomAgent("Agent3")]
        
        with pytest.raises(ValueError) as excinfo:
            MatchRunner(gomoku_game, agents)
        
        assert "requires 2 players" in str(excinfo.value)


class TestMatchRunnerCore:
    """Test the core match running functionality."""
    
    def test_run_match_basic(self, gomoku_game):
        """Test a basic match run with mocked agents."""
        # Create mock agents that make predefined moves
        agent1 = MagicMock()
        agent1.name = "MockAgent1"
        agent1.make_move.return_value = (0, 0)  # Always returns top-left corner
        
        agent2 = MagicMock()
        agent2.name = "MockAgent2"
        agent2.make_move.return_value = (0, 1)  # Always returns position to the right
        
        runner = MatchRunner(gomoku_game, [agent1, agent2])
        
        # Run a short match (limited to 5 moves)
        result = runner.run_match(max_moves=5, verbose=False)
        
        # Verify the result structure
        assert "match_id" in result
        assert result["game"] == "Gomoku"
        assert result["agents"] == ["MockAgent1", "MockAgent2"]
        assert result["moves"] <= 5
        assert "winner" in result
        assert "avg_move_time" in result
        assert "max_move_time" in result
        
        # Verify agents were called
        assert agent1.make_move.called
        assert agent2.make_move.called
        assert agent1.game_start.called
        assert agent2.game_start.called
        assert agent1.game_end.called
        assert agent2.game_end.called
    
    def test_agent_feedback_for_invalid_move(self):
        """Test that agents receive appropriate feedback for invalid moves."""
        # Create mock game and agents
        mock_game = MagicMock()
        mock_game.name = "MockGame"
        mock_game.num_players = 2
        
        # Configure mock game behavior
        mock_state = np.zeros((3, 3), dtype=np.int8)  # Simple 3x3 empty board
        mock_game.reset.return_value = mock_state
        mock_game.is_terminal.return_value = False  # Game not over
        mock_game.get_legal_moves.return_value = [(0, 0), (0, 1)]  # Legal moves
        mock_game.move_to_string.return_value = "A1"  # Simple move representation
        
        # Critical for JSON serialization during logging
        mock_game.get_winner.return_value = 0  # First player wins
        mock_game.get_state_representation.return_value = {"board": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]}  # JSON-serializable board
        mock_game.display_state.return_value = "Mock Board Display"  # String representation
        
        # The make_move method will raise an error for an invalid move
        def mock_make_move(state, move, player):
            if move == (9, 9):  # This is our invalid move
                raise ValueError("Invalid move")
            return mock_state  # Return unchanged state for simplicity
        
        mock_game.make_move.side_effect = mock_make_move
        
        # Create agents
        agent1 = MagicMock()
        agent1.name = "BadAgent"
        # Agent will make a bad move and then a good move when called again
        agent1.make_move.side_effect = [(9, 9), (0, 0)]
        
        agent2 = MagicMock()
        agent2.name = "GoodAgent"
        agent2.make_move.return_value = (0, 1)
        
        # Create runner with our mock objects
        runner = MatchRunner(mock_game, [agent1, agent2])
        
        # Patch the logger to prevent actual file writing
        with patch.object(runner.logger, 'log_match', return_value="mock_log_file.json"):
            # Run a short match
            result = runner.run_match(max_moves=3, verbose=False)
        
        # Verify the agent received feedback about the invalid move
        # We expect at least one call with success=False (invalid move)
        agent1.move_feedback.assert_called()
        feedback_calls = agent1.move_feedback.call_args_list
        
        # Find calls where success is False (invalid move feedback)
        invalid_move_feedbacks = [call for call in feedback_calls 
                                  if call[0][3] is False]  # success parameter is False
        
        assert len(invalid_move_feedbacks) > 0, "Agent should have received feedback about invalid move"
        
        # Verify that the match was able to complete despite the invalid move
        assert result is not None
        assert "moves" in result
        
        # Verify feedback was given for the invalid move
        assert agent1.move_feedback.called
        
        # Check that states_history contains the record of moves
        assert len(runner.states_history) > 0
    
    def test_run_match_to_win(self, small_gomoku_game):
        """Test running a match to a win condition."""
        # Create mock agents where one will win quickly
        agent1 = MagicMock()
        agent1.name = "WinningAgent"
        # Make moves that will create 3 in a row for win_length=3
        agent1.make_move.side_effect = [(0, 0), (0, 1), (0, 2)]
        
        agent2 = MagicMock()
        agent2.name = "LosingAgent"
        agent2.make_move.side_effect = [(1, 0), (1, 1)]
        
        runner = MatchRunner(small_gomoku_game, [agent1, agent2])
        
        result = runner.run_match(max_moves=10, verbose=False)
        
        # Verify the win
        assert result["winner"] == 0  # Player 0 (agent1) should win
        assert result["moves"] == 5  # 5 moves in total (3 by agent1, 2 by agent2)


class TestMatchRunnerLogging:
    """Test the logging functionality of MatchRunner."""
    
    def test_log_match(self, gomoku_game, random_agent):
        """Test that match results are logged correctly."""
        agents = [random_agent, RandomAgent("Agent2")]
        logger_mock = MagicMock()
        
        runner = MatchRunner(gomoku_game, agents, logger=logger_mock)
        
        # Create a simple result to log
        result = {
            "match_id": runner.match_id,
            "game": gomoku_game.name,
            "agents": [agent.name for agent in agents],
            "moves": 5,
            "winner": 0,
            "avg_move_time": 0.01,
            "max_move_time": 0.02
        }
        
        # Initialize states_history with a dummy state to prevent IndexError
        runner.states_history = [np.zeros((15, 15), dtype=np.int8)]
        
        # Log the match
        runner.log_match(result)
        
        # Verify the logger was called
        logger_mock.log_match.assert_called_once()
        
        # Check that the data passed to the logger contains the result
        log_data = logger_mock.log_match.call_args[0][0]
        assert log_data["match_id"] == result["match_id"]
        assert log_data["game"] == result["game"]
        assert log_data["agents"] == result["agents"]
        assert "moves_history" in log_data
        assert "final_state" in log_data
