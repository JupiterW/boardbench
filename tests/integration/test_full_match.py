"""
Integration tests for full game matches.

These tests validate that the entire system works together correctly,
running complete matches between different agents.
"""

import pytest
import os
import numpy as np
from unittest.mock import patch

from boardbench.games.gomoku import Gomoku
from boardbench.agents.random_agent import RandomAgent
from boardbench.engine.match_runner import MatchRunner
from boardbench.utils.logger import Logger


class TestRandomVsRandom:
    """Test full matches between random agents."""
    
    @pytest.fixture
    def log_dir(self, tmpdir):
        """Create a temporary log directory."""
        log_dir = os.path.join(str(tmpdir), "match_logs")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    def test_small_gomoku_match(self, log_dir):
        """Run a complete match on a small Gomoku board."""
        # Use a small board and short win length for faster tests
        game = Gomoku(board_size=5, win_length=3)
        
        # Create two random agents with fixed seeds for reproducibility
        agent1 = RandomAgent(name="TestRandom1", seed=42)
        agent2 = RandomAgent(name="TestRandom2", seed=24)
        
        # Create a logger with the temp directory
        logger = Logger(log_dir)
        
        # Create and run the match
        runner = MatchRunner(game, [agent1, agent2], logger=logger)
        result = runner.run_match(max_moves=25, verbose=False)
        
        # Verify the match completed successfully
        assert "match_id" in result
        assert result["game"] == "Gomoku"
        assert result["agents"] == ["TestRandom1", "TestRandom2"]
        assert result["moves"] > 0
        
        # Check that the moves history was recorded
        assert len(runner.moves_history) > 0
        
        # Check that states were recorded
        assert len(runner.states_history) > 0
        
        # Check that the log file was created
        log_files = os.listdir(log_dir)
        assert len(log_files) == 1
    
    def test_fixed_outcome_with_seeds(self, log_dir):
        """Test that with fixed seeds, the outcome is deterministic."""
        # Use a small board for faster tests
        game = Gomoku(board_size=5, win_length=3)
        
        # Run two matches with the same agents and seeds
        results = []
        
        for _ in range(2):
            # Reset the seeds for each match
            agent1 = RandomAgent(name="TestRandom1", seed=42)
            agent2 = RandomAgent(name="TestRandom2", seed=24)
            
            runner = MatchRunner(game, [agent1, agent2], logger=Logger(log_dir))
            result = runner.run_match(max_moves=25, verbose=False)
            results.append(result)
        
        # The results should be identical since we used the same seeds
        assert results[0]["moves"] == results[1]["moves"]
        assert results[0]["winner"] == results[1]["winner"]
        
        # The moves should be identical
        for move1, move2 in zip(results[0].get("moves_history", []), results[1].get("moves_history", [])):
            if "move" in move1 and "move" in move2:
                assert move1["move"] == move2["move"]
    
    def test_game_to_completion(self):
        """Test that a game runs to completion (win or draw)."""
        # Use a small board for faster tests
        game = Gomoku(board_size=5, win_length=3)
        
        # Create agents
        agent1 = RandomAgent(name="Player1")
        agent2 = RandomAgent(name="Player2")
        
        runner = MatchRunner(game, [agent1, agent2])
        result = runner.run_match(max_moves=25, verbose=False)
        
        # The game should be terminal by the end
        final_state = runner.states_history[-1]
        assert game.is_terminal(final_state) or result["moves"] >= 25
        
        # Winner should be reported correctly if there is one
        if game.get_winner(final_state) is not None:
            assert result["winner"] == game.get_winner(final_state)
        else:
            assert result["winner"] is None


class TestGameEngineIntegration:
    """Test the integration of game rules with the match engine."""
    
    def test_win_detection_integration(self):
        """Test that the engine correctly detects wins through the game rules."""
        # Create a small Gomoku game
        game = Gomoku(board_size=5, win_length=3)
        
        # Create mock agents that make specific moves to create a win
        class ScriptedAgent:
            def __init__(self, name, moves):
                self._name = name
                self.moves = moves
                self.move_index = 0
            
            @property
            def name(self):
                return self._name
                
            def make_move(self, game, state, legal_moves, player):
                if self.move_index < len(self.moves) and self.moves[self.move_index] in legal_moves:
                    move = self.moves[self.move_index]
                else:
                    # Fallback to first legal move
                    move = legal_moves[0] if legal_moves else None
                self.move_index += 1
                return move
                
            def game_start(self, game, player):
                pass
                
            def game_end(self, game, state, winner, player):
                pass
                
            def move_feedback(self, game, state, move, success, message):
                pass
        
        # Player 1 will make moves to win
        agent1_moves = [(0, 0), (1, 1), (2, 2)]  # Diagonal win
        agent1 = ScriptedAgent("WinningAgent", agent1_moves)
        
        # Player 2 makes arbitrary moves
        agent2_moves = [(0, 1), (1, 0)]
        agent2 = ScriptedAgent("LosingAgent", agent2_moves)
        
        runner = MatchRunner(game, [agent1, agent2])
        result = runner.run_match(verbose=False)
        
        # Verify player 1 won
        assert result["winner"] == 0
        assert result["moves"] == 5  # 3 by agent1, 2 by agent2
        
        # Verify the final state shows the win
        final_state = runner.states_history[-1]
        assert game.get_winner(final_state) == 0
        
        # Check that the diagonal line is formed
        assert final_state[0, 0] == 1  # Player 1's stone
        assert final_state[1, 1] == 1
        assert final_state[2, 2] == 1
