"""
Unit tests for the Logger utility.
"""

import pytest
import os
import json
import shutil
import numpy as np
from datetime import datetime
from unittest.mock import patch, mock_open

from boardbench.utils.logger import Logger


class TestLogger:
    """Test the Logger functionality."""
    
    @pytest.fixture
    def temp_log_dir(self, tmpdir):
        """Create a temporary log directory for testing."""
        log_dir = os.path.join(str(tmpdir), "test_logs")
        os.makedirs(log_dir, exist_ok=True)
        yield log_dir
        # Clean up
        shutil.rmtree(log_dir)
    
    def test_logger_initialization(self, temp_log_dir):
        """Test that the logger initializes and creates directory if needed."""
        # Remove the directory to test creation
        shutil.rmtree(temp_log_dir)
        assert not os.path.exists(temp_log_dir)
        
        # Initialize the logger which should create the directory
        logger = Logger(temp_log_dir)
        
        assert os.path.exists(temp_log_dir)
        assert logger.log_dir == temp_log_dir
    
    def test_log_match(self, temp_log_dir):
        """Test logging match data to a file."""
        logger = Logger(temp_log_dir)
        
        # Sample match data
        match_data = {
            "match_id": "test-match-123",
            "game": "Gomoku",
            "date": datetime.now().isoformat(),
            "agents": ["Agent1", "Agent2"],
            "moves": 25,
            "winner": 0,
            "avg_move_time": 0.05,
            "max_move_time": 0.1,
        }
        
        # Log the match
        log_path = logger.log_match(match_data)
        
        # Verify file exists
        assert os.path.exists(log_path)
        
        # Verify file content
        with open(log_path, 'r') as f:
            loaded_data = json.load(f)
            
            assert loaded_data["match_id"] == match_data["match_id"]
            assert loaded_data["game"] == match_data["game"]
            assert loaded_data["agents"] == match_data["agents"]
            assert loaded_data["moves"] == match_data["moves"]
            assert loaded_data["winner"] == match_data["winner"]
    
    def test_log_match_with_numpy_arrays(self, temp_log_dir):
        """Test logging match data that includes numpy arrays."""
        logger = Logger(temp_log_dir)
        
        # Sample match data with numpy array
        board = np.zeros((5, 5), dtype=np.int8)
        board[0, 0] = 1
        board[1, 1] = 2
        
        match_data = {
            "match_id": "test-numpy-123",
            "game": "Gomoku",
            "final_state": board,
            "moves_history": [
                {"board": board.copy()}
            ]
        }
        
        # Log the match
        log_path = logger.log_match(match_data)
        
        # Verify file exists and can be loaded
        assert os.path.exists(log_path)
        
        # Load and verify numpy arrays were converted properly
        with open(log_path, 'r') as f:
            loaded_data = json.load(f)
            
            assert loaded_data["match_id"] == match_data["match_id"]
            assert isinstance(loaded_data["final_state"], list)
            assert loaded_data["final_state"][0][0] == 1
            assert loaded_data["final_state"][1][1] == 2
            assert isinstance(loaded_data["moves_history"][0]["board"], list)
    
    def test_read_match(self, temp_log_dir):
        """Test reading match data from a file."""
        logger = Logger(temp_log_dir)
        
        # Create sample match data
        sample_data = {
            "match_id": "test-read-123",
            "game": "Gomoku",
            "winner": 1
        }
        
        # Write it to a file
        file_path = os.path.join(temp_log_dir, "test_match.json")
        with open(file_path, 'w') as f:
            json.dump(sample_data, f)
        
        # Read it back
        loaded_data = logger.read_match(file_path)
        
        assert loaded_data["match_id"] == sample_data["match_id"]
        assert loaded_data["game"] == sample_data["game"]
        assert loaded_data["winner"] == sample_data["winner"]
    
    def test_read_match_nonexistent_file(self, temp_log_dir):
        """Test reading from a nonexistent file raises appropriate error."""
        logger = Logger(temp_log_dir)
        
        nonexistent_file = os.path.join(temp_log_dir, "nonexistent.json")
        
        with pytest.raises(FileNotFoundError):
            logger.read_match(nonexistent_file)
