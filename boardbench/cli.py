import os
import sys
import typer
from typing import List, Optional
from rich.console import Console
from rich.table import Table

from boardbench.engine.match_runner import MatchRunner
from boardbench.games.connect4 import Connect4
from boardbench.games.gomoku import Gomoku
from boardbench.agents.random_agent import RandomAgent
from boardbench.agents.enforced_random_agent import EnforcedRandomAgent
from boardbench.agents.llm_agent import LLMAgent


# Create console for rich output
console = Console()

# Create Typer app
app = typer.Typer(help="BoardBench: Board Game Benchmarking Framework")


@app.command()
def run(
    game_name: str = typer.Option("gomoku", "--game", "-g", help="Game to play: gomoku or connect4"),
    agent1: str = typer.Option("random", "--agent1", "-a1", help="First agent type: random, enforced-random, or llm"),
    agent2: str = typer.Option("random", "--agent2", "-a2", help="Second agent type: random, enforced-random, or llm"),
    model: str = typer.Option("gpt-4", "--model", "-m", help="Model name for LLM agent(s)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="OpenAI API key for LLM agent(s)"),
    max_moves: int = typer.Option(1000, "--max-moves", help="Maximum number of moves before declaring a draw"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Display game progress"),
):
    """Run a match between two agents."""
    # Select the game
    game = None
    if game_name.lower() == "gomoku":
        game = Gomoku()
    elif game_name.lower() == "connect4":
        game = Connect4()
    else:
        console.print(f"[red]Unknown game: {game_name}[/red]")
        sys.exit(1)
    
    console.print(f"[green]Running a {game.name} match...[/green]")
    
    # Create agents
    agents = []
    agent_configs = [(agent1, "Player 1"), (agent2, "Player 2")]
    
    for agent_type, agent_name in agent_configs:
        if agent_type.lower() == "random":
            agents.append(RandomAgent(f"Random {agent_name}"))
        elif agent_type.lower() == "enforced-random":
            agents.append(EnforcedRandomAgent(f"EnforcedRandom {agent_name}"))
        elif agent_type.lower() == "llm":
            try:
                agents.append(LLMAgent(
                    name=f"LLM {agent_name}",
                    model=model,
                    api_key=api_key
                ))
            except ImportError:
                console.print("[red]OpenAI package is required for LLM agents. Install with: pip install openai[/red]")
                sys.exit(1)
        else:
            console.print(f"[red]Unknown agent type: {agent_type}[/red]")
            sys.exit(1)
    
    # Create and run the match
    try:
        match_runner = MatchRunner(game, agents)
        result = match_runner.run_match(max_moves=max_moves, verbose=verbose)
        
        # Display results
        console.print("\n[bold]Match Results[/bold]")
        
        table = Table(show_header=True)
        table.add_column("Property")
        table.add_column("Value")
        
        table.add_row("Game", result["game"])
        table.add_row("Match ID", result["match_id"])
        table.add_row("Date", result["date"])
        table.add_row("Player 1", result["agents"][0])
        table.add_row("Player 2", result["agents"][1])
        table.add_row("Total Moves", str(result["moves"]))
        
        if result["winner"] is not None:
            winner_name = result["agents"][result["winner"]]
            table.add_row("Winner", f"Player {result['winner'] + 1} ({winner_name})")
        else:
            table.add_row("Winner", "Draw")
            
        table.add_row("Average Move Time", f"{result['avg_move_time']:.3f}s")
        table.add_row("Maximum Move Time", f"{result['max_move_time']:.3f}s")
        
        console.print(table)
        console.print(f"\nMatch log saved to: {match_runner.logger.log_dir}\n")
        
    except Exception as e:
        console.print(f"[red]Error during match: {str(e)}[/red]")
        sys.exit(1)


@app.command()
def list_games():
    """List available games."""
    table = Table(show_header=True)
    table.add_column("Game")
    table.add_column("Description")
    
    table.add_row("gomoku", "Connect 5 stones in a row on a 15x15 grid")
    table.add_row("connect4", "Connect 4 pieces in a row in a 6x7 grid")
    
    console.print(table)


@app.command()
def list_agents():
    """List available agents."""
    table = Table(show_header=True)
    table.add_column("Agent")
    table.add_column("Description")
    table.add_column("Requirements")
    
    table.add_row("random", "Makes random legal moves", "None")
    table.add_row("enforced-random", "Makes random legal moves with extra validation to prevent invalid moves", "None")
    table.add_row("llm", "Uses OpenAI's API to make decisions", "openai package, API key")
    
    console.print(table)


if __name__ == "__main__":
    app()
