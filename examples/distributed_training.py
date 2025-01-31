"""Example of distributed training with safety checks in LLM GYM."""

import os
import sys
from typing import Dict, Any
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_gym.envs import (
    LogicPuzzleEnv,
    DialogueEnv,
    MemoryBufferEnv
)
from llm_gym.distributed import DistributedTrainer
from llm_gym.safety import SafetyManager
from llm_gym.monitoring.visualization import (
    MetricsLogger,
    PerformanceVisualizer
)

def setup_environments():
    """Set up environment configurations."""
    env_configs = {
        "logic_puzzle": {
            "class": LogicPuzzleEnv,
            "config": {
                "max_steps": 10,
                "num_premises": 3,
                "difficulty": 0.5
            }
        },
        "dialogue": {
            "class": DialogueEnv,
            "config": {
                "max_turns": 5,
                "complexity": 0.5
            }
        },
        "memory": {
            "class": MemoryBufferEnv,
            "config": {
                "buffer_size": 5,
                "max_steps": 8
            }
        }
    }
    return env_configs

def run_distributed_training(
    env_configs: Dict[str, Any],
    num_workers: int = 4,
    episodes_per_env: int = 25
):
    """Run distributed training across multiple environments.
    
    Args:
        env_configs (Dict[str, Any]): Environment configurations
        num_workers (int): Number of parallel workers
        episodes_per_env (int): Episodes to run per environment
    """
    safety_manager = SafetyManager(["content_safety", "fairness", "privacy"])
    metrics_logger = MetricsLogger()
    visualizer = PerformanceVisualizer(metrics_logger)
    
    print(f"Starting distributed training with {num_workers} workers...")
    print(f"Running {episodes_per_env} episodes per environment type")
    
    for env_name, env_spec in env_configs.items():
        print(f"\nTraining on {env_name} environment...")
        
        # Initialize distributed trainer
        trainer = DistributedTrainer(
            num_workers=num_workers,
            env_class=env_spec["class"],
            env_config=env_spec["config"]
        )
        
        # Run episodes in parallel
        results = trainer.parallel_rollout(episodes_per_env)
        
        # Process results and check safety
        safe_episodes = 0
        total_reward = 0
        total_steps = 0
        
        for episode in results:
            # Check safety of each action in episode
            episode_safe = True
            for step in episode["steps"]:
                is_safe, _ = safety_manager.check_action(step["action"])
                if not is_safe:
                    episode_safe = False
                    break
                    
            if episode_safe:
                safe_episodes += 1
            total_reward += episode["total_reward"]
            total_steps += episode["num_steps"]
            
        # Log aggregate metrics
        print(f"Results for {env_name}:")
        print(f"Safe episodes: {safe_episodes}/{episodes_per_env}")
        print(f"Average reward: {total_reward/episodes_per_env:.2f}")
        print(f"Average steps: {total_steps/episodes_per_env:.2f}")
        
        # Generate visualizations
        visualizer.plot_learning_curve(
            save_path=f"reports/{env_name}_learning_curve.png"
        )
        visualizer.plot_safety_violations(
            save_path=f"reports/{env_name}_safety_violations.png"
        )
        
        # Cleanup
        trainer.shutdown()
    
    # Generate final report
    print("\nGenerating final training report...")
    visualizer.generate_report(
        output_dir="reports",
        prefix="distributed_training"
    )
    print("Training complete! Reports available in 'reports' directory")

def main():
    """Run distributed training example."""
    env_configs = setup_environments()
    run_distributed_training(env_configs)

if __name__ == "__main__":
    main() 