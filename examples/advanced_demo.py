"""Advanced demo showcasing LLM GYM features."""

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
from llm_gym.safety import SafetyManager
from llm_gym.curriculum.enhanced import EnhancedCurriculum
from llm_gym.monitoring.visualization import (
    MetricsLogger,
    PerformanceVisualizer,
    TrainingMetrics
)
from llm_gym.rewards.adaptive import (
    AdaptiveRewardManager,
    RewardComponent,
    TaskCompletionReward,
    EfficiencyReward,
    SafetyReward,
    NoveltyReward
)

def setup_reward_manager() -> AdaptiveRewardManager:
    """Set up adaptive reward manager with components."""
    components = [
        RewardComponent(
            name="task_completion",
            weight=0.4,
            min_value=0.0,
            max_value=1.0,
            threshold=0.5,
            is_binary=True
        ),
        RewardComponent(
            name="efficiency",
            weight=0.3,
            min_value=0.0,
            max_value=1.0,
            threshold=0.7,
            is_binary=False
        ),
        RewardComponent(
            name="safety",
            weight=0.2,
            min_value=-1.0,
            max_value=0.0,
            threshold=0.0,
            is_binary=True
        ),
        RewardComponent(
            name="novelty",
            weight=0.1,
            min_value=0.0,
            max_value=1.0,
            threshold=0.3,
            is_binary=False
        )
    ]
    
    return AdaptiveRewardManager(components)

def setup_curriculum() -> EnhancedCurriculum:
    """Set up enhanced curriculum with task templates."""
    task_templates = {
        "meta_reasoning": {
            "parameters": {
                "max_steps": 10,
                "complexity": 0.5,
                "required_skills": ["planning", "evaluation"]
            }
        },
        "logic_puzzle": {
            "parameters": {
                "num_premises": 3,
                "num_relations": 2,
                "max_steps": 5
            }
        },
        "dialogue": {
            "parameters": {
                "max_turns": 5,
                "required_elements": ["greeting", "query", "response"],
                "complexity": 0.5
            }
        },
        "memory": {
            "parameters": {
                "buffer_size": 5,
                "num_items": 3,
                "max_steps": 8
            }
        }
    }
    
    return EnhancedCurriculum(task_templates)

def simulate_episode(
    env: Any,
    reward_manager: AdaptiveRewardManager,
    safety_manager: SafetyManager,
    metrics_logger: MetricsLogger
) -> Dict[str, float]:
    """Simulate one training episode.
    
    Args:
        env: Environment instance
        reward_manager: Reward manager instance
        safety_manager: Safety manager instance
        metrics_logger: Metrics logger instance
        
    Returns:
        Dict[str, float]: Episode metrics
    """
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    violations = 0
    
    while not done:
        # Simulate agent action (replace with actual agent)
        action = "SIMULATE_ACTION"  # Placeholder
        
        # Check safety
        is_safe, safety_info = safety_manager.check_action(action)
        if not is_safe:
            violations += 1
            
        # Take step
        next_obs, reward, done, info = env.step(action)
        
        # Compute adaptive reward
        state = {
            "task_completed": done,
            "steps_taken": steps,
            "max_steps": env.max_steps,
            "safety_violations": violations,
            "solution": action
        }
        
        adaptive_reward, component_values = reward_manager.compute_reward(
            state, action
        )
        
        total_reward += adaptive_reward
        steps += 1
        obs = next_obs
        
    # Log metrics
    metrics = TrainingMetrics(
        timestamp=datetime.now().timestamp(),
        episode=env.episode_count,
        reward=total_reward,
        success=done and total_reward > 0,
        task_type=env.__class__.__name__,
        difficulty=env.difficulty if hasattr(env, "difficulty") else 0.5,
        completion_time=steps,
        policy_violations=violations
    )
    
    metrics_logger.log_metrics(metrics)
    
    return {
        "reward": total_reward,
        "steps": steps,
        "violations": violations,
        "success": done and total_reward > 0
    }

def main():
    """Run advanced demo."""
    # Set up components
    reward_manager = setup_reward_manager()
    curriculum = setup_curriculum()
    safety_manager = SafetyManager()
    metrics_logger = MetricsLogger()
    visualizer = PerformanceVisualizer(metrics_logger)
    
    # Initialize environments
    envs = {
        "meta_reasoning": MetaReasoningEnv(),
        "logic_puzzle": LogicPuzzleEnv(),
        "dialogue": DialogueEnv(),
        "memory": MemoryBufferEnv()
    }
    
    num_episodes = 100
    print(f"Running {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Generate task from curriculum
        task = curriculum.generate_task()
        env = envs[task["id"]]
        
        # Run episode
        metrics = simulate_episode(
            env,
            reward_manager,
            safety_manager,
            metrics_logger
        )
        
        # Update curriculum
        curriculum.complete_task(
            success=metrics["success"],
            reward=metrics["reward"],
            attempts=metrics["steps"]
        )
        
        # Update reward weights
        performance = {
            "task_completion": float(metrics["success"]),
            "efficiency": 1.0 - (metrics["steps"] / env.max_steps),
            "safety": float(metrics["violations"] == 0),
            "novelty": 0.5  # Placeholder
        }
        
        target = {
            "task_completion": 0.8,
            "efficiency": 0.7,
            "safety": 1.0,
            "novelty": 0.3
        }
        
        reward_manager.update_weights(performance, target)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Task: {task['id']}")
            print(f"Difficulty: {task['difficulty']:.2f}")
            print(f"Reward: {metrics['reward']:.2f}")
            print(f"Steps: {metrics['steps']}")
            print(f"Violations: {metrics['violations']}")
            print("Current weights:", reward_manager.get_component_weights())
            print()
    
    # Generate training report
    print("\nGenerating training report...")
    visualizer.generate_report()
    print("Report generated in 'reports' directory")

if __name__ == "__main__":
    main() 