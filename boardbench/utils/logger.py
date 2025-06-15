import os
import json
from datetime import datetime
from typing import Dict, Any, Optional


class Logger:
    """
    Logger for board game matches.
    
    Handles saving match data to JSON files for later analysis.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize the logger.
        
        Args:
            log_dir: The directory where logs will be saved
        """
        self.log_dir = log_dir
        
        # Create the log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def log_match(self, match_data: Dict[str, Any]) -> str:
        """
        Log a match to a JSON file.
        
        Args:
            match_data: Dictionary containing match data
            
        Returns:
            The path to the log file
        """
        # Generate a filename based on the match ID and date
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_name = match_data.get("game", "unknown")
        match_id = match_data.get("match_id", "unknown")
        
        filename = f"{timestamp}_{game_name}_{match_id}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        # This function recursively processes dictionaries and lists
        def prepare_for_json(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: prepare_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [prepare_for_json(item) for item in obj]
            elif hasattr(obj, 'tolist'):  # Handle numpy arrays
                return obj.tolist()
            else:
                return obj
        
        # Prepare the data for serialization
        json_data = prepare_for_json(match_data)
        
        # Write to the file
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return filepath
    
    def read_match(self, filepath: str) -> Dict[str, Any]:
        """
        Read a match log from a file.
        
        Args:
            filepath: Path to the log file
            
        Returns:
            Dictionary containing the match data
        """
        with open(filepath, 'r') as f:
            match_data = json.load(f)
        
        return match_data
