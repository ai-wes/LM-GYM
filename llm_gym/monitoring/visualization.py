"""Visualization and monitoring module for LLM GYM."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
import json
import os

@dataclass
class TrainingMetrics:
    """Training metrics for visualization."""
    timestamp: float
    episode: int
    reward: float
    success: bool
    task_type: str
    difficulty: float
    completion_time: float
    policy_violations: int

class MetricsLogger:
    """Logs training metrics for visualization."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize metrics logger.
        
        Args:
            log_dir (str): Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = os.path.join(log_dir, f"metrics_{self.current_session}.jsonl")
        self.metrics_buffer = []
        
    def log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics.
        
        Args:
            metrics (TrainingMetrics): Metrics to log
        """
        metrics_dict = {
            "timestamp": metrics.timestamp,
            "episode": metrics.episode,
            "reward": metrics.reward,
            "success": metrics.success,
            "task_type": metrics.task_type,
            "difficulty": metrics.difficulty,
            "completion_time": metrics.completion_time,
            "policy_violations": metrics.policy_violations
        }
        
        self.metrics_buffer.append(metrics_dict)
        
        # Write to file periodically
        if len(self.metrics_buffer) >= 100:
            self.flush()
            
    def flush(self):
        """Write buffered metrics to file."""
        if not self.metrics_buffer:
            return
            
        with open(self.metrics_file, "a") as f:
            for metrics in self.metrics_buffer:
                f.write(json.dumps(metrics) + "\n")
        
        self.metrics_buffer.clear()
        
    def load_metrics(self) -> List[Dict[str, Any]]:
        """Load all metrics from file.
        
        Returns:
            List[Dict[str, Any]]: List of metrics dictionaries
        """
        metrics = []
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, "r") as f:
                for line in f:
                    metrics.append(json.loads(line.strip()))
        return metrics

class PerformanceVisualizer:
    """Visualizes agent performance metrics."""
    
    def __init__(self, metrics_logger: MetricsLogger):
        """Initialize visualizer.
        
        Args:
            metrics_logger (MetricsLogger): Logger containing metrics
        """
        self.logger = metrics_logger
        
    def plot_learning_curve(self, 
                           window_size: int = 100,
                           save_path: Optional[str] = None):
        """Plot learning curve showing reward over time.
        
        Args:
            window_size (int): Window size for moving average
            save_path (Optional[str]): Path to save plot
        """
        metrics = self.logger.load_metrics()
        if not metrics:
            return
            
        episodes = [m["episode"] for m in metrics]
        rewards = [m["reward"] for m in metrics]
        
        # Calculate moving average
        window = np.ones(window_size) / window_size
        smoothed_rewards = np.convolve(rewards, window, mode="valid")
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes[window_size-1:], smoothed_rewards)
        plt.title("Learning Curve")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_task_distribution(self, save_path: Optional[str] = None):
        """Plot distribution of task types and success rates.
        
        Args:
            save_path (Optional[str]): Path to save plot
        """
        metrics = self.logger.load_metrics()
        if not metrics:
            return
            
        task_types = {}
        for m in metrics:
            task_type = m["task_type"]
            if task_type not in task_types:
                task_types[task_type] = {"total": 0, "success": 0}
            
            task_types[task_type]["total"] += 1
            if m["success"]:
                task_types[task_type]["success"] += 1
                
        types = list(task_types.keys())
        success_rates = [
            task_types[t]["success"] / task_types[t]["total"]
            for t in types
        ]
        
        plt.figure(figsize=(10, 6))
        plt.bar(types, success_rates)
        plt.title("Task Success Rates")
        plt.xlabel("Task Type")
        plt.ylabel("Success Rate")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_difficulty_progression(self, 
                                  window_size: int = 100,
                                  save_path: Optional[str] = None):
        """Plot difficulty progression over time.
        
        Args:
            window_size (int): Window size for moving average
            save_path (Optional[str]): Path to save plot
        """
        metrics = self.logger.load_metrics()
        if not metrics:
            return
            
        episodes = [m["episode"] for m in metrics]
        difficulties = [m["difficulty"] for m in metrics]
        
        # Calculate moving average
        window = np.ones(window_size) / window_size
        smoothed_difficulties = np.convolve(difficulties, window, mode="valid")
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes[window_size-1:], smoothed_difficulties)
        plt.title("Difficulty Progression")
        plt.xlabel("Episode")
        plt.ylabel("Average Difficulty")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_safety_violations(self, 
                             window_size: int = 100,
                             save_path: Optional[str] = None):
        """Plot safety policy violations over time.
        
        Args:
            window_size (int): Window size for moving average
            save_path (Optional[str]): Path to save plot
        """
        metrics = self.logger.load_metrics()
        if not metrics:
            return
            
        episodes = [m["episode"] for m in metrics]
        violations = [m["policy_violations"] for m in metrics]
        
        # Calculate moving average
        window = np.ones(window_size) / window_size
        smoothed_violations = np.convolve(violations, window, mode="valid")
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes[window_size-1:], smoothed_violations)
        plt.title("Safety Policy Violations")
        plt.xlabel("Episode")
        plt.ylabel("Average Violations")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def generate_report(self, 
                       output_dir: str = "reports",
                       prefix: str = "training_report"):
        """Generate comprehensive training report with all plots.
        
        Args:
            output_dir (str): Directory to save report
            prefix (str): Prefix for report files
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate all plots
        self.plot_learning_curve(
            save_path=os.path.join(output_dir, f"{prefix}_learning_curve_{timestamp}.png")
        )
        self.plot_task_distribution(
            save_path=os.path.join(output_dir, f"{prefix}_task_dist_{timestamp}.png")
        )
        self.plot_difficulty_progression(
            save_path=os.path.join(output_dir, f"{prefix}_difficulty_{timestamp}.png")
        )
        self.plot_safety_violations(
            save_path=os.path.join(output_dir, f"{prefix}_violations_{timestamp}.png")
        )
        
        # Generate summary statistics
        metrics = self.logger.load_metrics()
        if not metrics:
            return
            
        summary = {
            "total_episodes": len(metrics),
            "avg_reward": np.mean([m["reward"] for m in metrics]),
            "success_rate": np.mean([1 if m["success"] else 0 for m in metrics]),
            "avg_completion_time": np.mean([m["completion_time"] for m in metrics]),
            "total_violations": sum(m["policy_violations"] for m in metrics)
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, f"{prefix}_summary_{timestamp}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2) 