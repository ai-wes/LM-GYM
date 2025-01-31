"""Quickstart demo using Qwen2.5-3B-Instruct model with LLM GYM."""

import os
import sys
from typing import Dict, Any
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_gym.envs import (
    MetaReasoningEnv,
    LogicPuzzleEnv,
    DialogueEnv,
    MemoryBufferEnv
)
from llm_gym.quickstart import QwenAgent
from llm_gym.safety import SafetyManager
from llm_gym.monitoring.visualization import (
    MetricsLogger,
    PerformanceVisualizer,
    TrainingMetrics
)

def run_environment_demo(
    env_name: str,
    env,
    agent: QwenAgent,
    safety_manager: SafetyManager,
    metrics_logger: MetricsLogger,
    num_episodes: int = 5
) -> None:
    """Run demo episodes for a specific environment.
    
    Args:
        env_name (str): Name of the environment
        env: Environment instance
        agent (QwenAgent): Qwen agent instance
        safety_manager (SafetyManager): Safety manager instance
        metrics_logger (MetricsLogger): Metrics logger instance
        num_episodes (int): Number of episodes to run
    """
    print(f"\nRunning {num_episodes} episodes on {env_name}...")
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        violations = 0
        
        while not done:
            # Get action from agent
            action = agent.act(obs)
            print(f"\nStep {steps + 1}:")
            print(f"Observation: {obs}")
            print(f"Action: {action}")
            
            # Check safety
            is_safe, safety_info = safety_manager.check_action(action)
            if not is_safe:
                print("Safety violation detected!")
                print(f"Details: {safety_info}")
                violations += 1
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            print(f"Reward: {reward}")
            
            total_reward += reward
            steps += 1
            obs = next_obs
            
        # Log episode metrics
        metrics = TrainingMetrics(
            timestamp=datetime.now().timestamp(),
            episode=episode,
            reward=total_reward,
            success=done and total_reward > 0,
            task_type=env_name,
            difficulty=getattr(env, "difficulty", 0.5),
            completion_time=steps,
            policy_violations=violations
        )
        metrics_logger.log_metrics(metrics)
        
        print(f"\nEpisode complete!")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Steps taken: {steps}")
        print(f"Safety violations: {violations}")

def main():
    """Run quickstart demo with Qwen agent."""
    # Initialize components
    agent = QwenAgent()  # Will use CUDA if available
    safety_manager = SafetyManager(["content_safety", "fairness"])
    metrics_logger = MetricsLogger()
    visualizer = PerformanceVisualizer(metrics_logger)
    
    # Initialize environments
    environments = {
        "meta_reasoning": MetaReasoningEnv(max_steps=10),
        "logic_puzzle": LogicPuzzleEnv(max_steps=5),
        "dialogue": DialogueEnv(max_turns=5),
        "memory": MemoryBufferEnv(max_steps=8)
    }
    
    # Run demo for each environment
    for env_name, env in environments.items():
        run_environment_demo(
            env_name=env_name,
            env=env,
            agent=agent,
            safety_manager=safety_manager,
            metrics_logger=metrics_logger
        )
        
        # Generate environment-specific plots
        visualizer.plot_learning_curve(
            save_path=f"reports/quickstart_{env_name}_learning.png"
        )
        visualizer.plot_safety_violations(
            save_path=f"reports/quickstart_{env_name}_safety.png"
        )
    
    # Generate comprehensive report
    print("\nGenerating training report...")
    visualizer.generate_report(
        output_dir="reports",
        prefix="quickstart_demo"
    )
    print("Demo complete! Reports available in 'reports' directory")

if __name__ == "__main__":
    main() 