"""Example of transfer learning across different LLM GYM environments."""

import os
import sys
from typing import Dict, Any, List
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
from llm_gym.curriculum.enhanced import EnhancedCurriculum
from llm_gym.monitoring.visualization import (
    MetricsLogger,
    PerformanceVisualizer,
    TrainingMetrics
)

def setup_curriculum() -> EnhancedCurriculum:
    """Set up curriculum for transfer learning."""
    task_templates = {
        "meta_reasoning": {
            "parameters": {
                "max_steps": 10,
                "complexity": 0.3,  # Start easier
                "required_skills": ["planning"]
            }
        },
        "logic_puzzle": {
            "parameters": {
                "num_premises": 2,  # Start with simpler puzzles
                "num_relations": 1,
                "max_steps": 5
            }
        },
        "dialogue": {
            "parameters": {
                "max_turns": 3,  # Start with shorter dialogues
                "required_elements": ["greeting", "response"],
                "complexity": 0.3
            }
        },
        "memory": {
            "parameters": {
                "buffer_size": 3,  # Start with smaller memory
                "num_items": 2,
                "max_steps": 5
            }
        }
    }
    
    return EnhancedCurriculum(task_templates)

def train_environment(
    env_name: str,
    env,
    agent: QwenAgent,
    safety_manager: SafetyManager,
    metrics_logger: MetricsLogger,
    num_episodes: int = 10
) -> List[float]:
    """Train agent on specific environment.
    
    Args:
        env_name (str): Name of the environment
        env: Environment instance
        agent (QwenAgent): Qwen agent instance
        safety_manager (SafetyManager): Safety manager instance
        metrics_logger (MetricsLogger): Metrics logger instance
        num_episodes (int): Number of episodes to run
        
    Returns:
        List[float]: Episode rewards
    """
    print(f"\nTraining on {env_name} environment...")
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        violations = 0
        
        while not done:
            # Get action from agent
            action = agent.act(obs)
            
            # Check safety
            is_safe, safety_info = safety_manager.check_action(action)
            if not is_safe:
                violations += 1
                print(f"Safety violation: {safety_info}")
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            obs = next_obs
        
        # Log metrics
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
        
        episode_rewards.append(total_reward)
        print(f"Episode reward: {total_reward:.2f}")
    
    return episode_rewards

def evaluate_transfer(
    source_env_name: str,
    target_env_name: str,
    source_rewards: List[float],
    target_rewards: List[float]
) -> None:
    """Evaluate transfer learning performance.
    
    Args:
        source_env_name (str): Name of source environment
        target_env_name (str): Name of target environment
        source_rewards (List[float]): Rewards from source environment
        target_rewards (List[float]): Rewards from target environment
    """
    print(f"\nTransfer Learning Analysis: {source_env_name} -> {target_env_name}")
    print(f"Source environment average reward: {np.mean(source_rewards):.2f}")
    print(f"Target environment average reward: {np.mean(target_rewards):.2f}")
    
    # Compare learning speeds
    source_learning_rate = (source_rewards[-1] - source_rewards[0]) / len(source_rewards)
    target_learning_rate = (target_rewards[-1] - target_rewards[0]) / len(target_rewards)
    
    print(f"Source learning rate: {source_learning_rate:.3f}")
    print(f"Target learning rate: {target_learning_rate:.3f}")
    
    # Check for positive transfer
    if target_learning_rate > source_learning_rate:
        print("Positive transfer detected: Faster learning in target environment")
    elif target_learning_rate < source_learning_rate:
        print("Negative transfer detected: Slower learning in target environment")
    else:
        print("Neutral transfer: Similar learning rates")

def main():
    """Run transfer learning experiment."""
    # Initialize components
    agent = QwenAgent()
    safety_manager = SafetyManager(["content_safety", "fairness"])
    metrics_logger = MetricsLogger()
    visualizer = PerformanceVisualizer(metrics_logger)
    curriculum = setup_curriculum()
    
    # Define environment sequence for transfer learning
    env_sequence = [
        ("meta_reasoning", MetaReasoningEnv),
        ("logic_puzzle", LogicPuzzleEnv),
        ("dialogue", DialogueEnv),
        ("memory", MemoryBufferEnv)
    ]
    
    # Train on each environment and evaluate transfer
    previous_rewards = None
    previous_env_name = None
    
    for env_name, env_class in env_sequence:
        # Get task parameters from curriculum
        task = curriculum.generate_task()
        env_config = task["parameters"]
        
        # Initialize environment
        env = env_class(**env_config)
        
        # Train and get rewards
        current_rewards = train_environment(
            env_name=env_name,
            env=env,
            agent=agent,
            safety_manager=safety_manager,
            metrics_logger=metrics_logger
        )
        
        # Evaluate transfer if not first environment
        if previous_rewards is not None:
            evaluate_transfer(
                source_env_name=previous_env_name,
                target_env_name=env_name,
                source_rewards=previous_rewards,
                target_rewards=current_rewards
            )
        
        # Generate visualizations
        visualizer.plot_learning_curve(
            save_path=f"reports/transfer_{env_name}_learning.png"
        )
        visualizer.plot_safety_violations(
            save_path=f"reports/transfer_{env_name}_safety.png"
        )
        
        # Update previous rewards
        previous_rewards = current_rewards
        previous_env_name = env_name
    
    # Generate final report
    print("\nGenerating transfer learning report...")
    visualizer.generate_report(
        output_dir="reports",
        prefix="transfer_learning"
    )
    print("Experiment complete! Reports available in 'reports' directory")

if __name__ == "__main__":
    main() 