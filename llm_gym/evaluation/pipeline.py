from typing import Any, Dict, List, Optional
import pandas as pd
from datetime import datetime
import json
import os
from tqdm import tqdm

class EvaluationPipeline:
    """Pipeline for evaluating LLM performance in environments."""
    
    def __init__(self, env, log_dir: str = "logs"):
        """Initialize evaluation pipeline.
        
        Args:
            env: The environment to evaluate
            log_dir (str): Directory to store logs
        """
        self.env = env
        self.log_dir = log_dir
        self.metrics = []
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging directory and files."""
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"eval_{self.current_time}.jsonl")
        
    def _log_episode(self, episode_data: Dict[str, Any]):
        """Log episode data to file.
        
        Args:
            episode_data (Dict[str, Any]): Data from one episode
        """
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(episode_data) + '\n')
            
    def evaluate(self, num_episodes: int = 100) -> pd.DataFrame:
        """Run evaluation for specified number of episodes.
        
        Args:
            num_episodes (int): Number of episodes to run
            
        Returns:
            pd.DataFrame: Evaluation results
        """
        episode_data = []
        
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            # Reset environment
            obs = self.env.reset()
            done = False
            total_reward = 0
            steps = []
            
            # Run episode
            while not done:
                # In practice, you would have an agent here making decisions
                # For now, we'll just use a dummy action
                action = "dummy_action"
                
                # Take step in environment
                next_obs, reward, done, info = self.env.step(action)
                
                # Track step data
                step_data = {
                    "observation": obs,
                    "action": action,
                    "reward": reward,
                    "next_observation": next_obs,
                    "done": done,
                    "info": info
                }
                steps.append(step_data)
                
                total_reward += reward
                obs = next_obs
            
            # Compile episode metrics
            episode_metrics = {
                "episode": episode,
                "total_reward": total_reward,
                "num_steps": len(steps),
                "timestamp": datetime.now().isoformat(),
                "steps": steps
            }
            
            # Add chain-of-thought metrics if available
            if hasattr(self.env, "captured_chain_of_thought"):
                episode_metrics["chain_of_thought_length"] = len(self.env.captured_chain_of_thought.split())
                
            episode_data.append(episode_metrics)
            self._log_episode(episode_metrics)
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame(episode_data)
        self._compute_aggregate_metrics(df)
        return df
    
    def _compute_aggregate_metrics(self, df: pd.DataFrame):
        """Compute aggregate metrics across all episodes.
        
        Args:
            df (pd.DataFrame): Episode data
        """
        metrics = {
            "mean_reward": df["total_reward"].mean(),
            "std_reward": df["total_reward"].std(),
            "mean_episode_length": df["num_steps"].mean(),
            "total_episodes": len(df)
        }
        
        if "chain_of_thought_length" in df.columns:
            metrics.update({
                "mean_chain_length": df["chain_of_thought_length"].mean(),
                "max_chain_length": df["chain_of_thought_length"].max()
            })
            
        self.metrics.append({
            "timestamp": datetime.now().isoformat(),
            **metrics
        })
        
    def plot_learning_curve(self, save_path: Optional[str] = None):
        """Plot learning curve from evaluation results.
        
        Args:
            save_path (Optional[str]): Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            df = pd.read_json(self.log_file, lines=True)
            plt.figure(figsize=(10, 6))
            plt.plot(df["episode"], df["total_reward"], label="Total Reward")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Learning Curve")
            plt.legend()
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except ImportError:
            print("matplotlib is required for plotting") 