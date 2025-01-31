"""
Demo script showing how to use the meta-reasoning environment.
This simulates an LLM agent interacting with the environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_gym.envs.meta_reasoning import MetaReasoningEnv
from llm_gym.evaluation import EvaluationPipeline

def simulate_llm_response(observation):
    """
    Simulate an LLM's response to an observation.
    In practice, this would be replaced with actual LLM calls.
    
    Args:
        observation: Environment observation
        
    Returns:
        str: Simulated LLM action
    """
    task = observation.get("task", {})
    reflection_types = task.get("reflection_types", [])
    
    if not observation.get("chain_of_thought"):
        # First step - provide reasoning
        reasoning = "Let me think about this step by step:\n"
        for r_type in reflection_types:
            reasoning += f"1. For {r_type}, I need to consider...\n"
        if task.get("uncertainty_tracking", False):
            reasoning += "2. I should express uncertainty where appropriate.\n"
        if task.get("required_justification", False):
            reasoning += "3. I need to justify my conclusions.\n"
        return f"THINK {reasoning}"
    else:
        # Provide final answer with required elements
        answer = "Based on my analysis:\n"
        for r_type in reflection_types:
            answer += f"Regarding {r_type}, I conclude that...\n"
        if task.get("uncertainty_tracking", False):
            answer += "I'm uncertain about... but I might...\n"
        if task.get("required_justification", False):
            answer += "Therefore, because of the above reasons...\n"
        return f"ANSWER {answer}"

def main():
    # Create environment
    env = MetaReasoningEnv(
        use_curriculum=True,
        use_human_feedback=False,
        max_steps=5
    )
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(env)
    
    # Run a few episodes
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}")
        print("-" * 50)
        
        # Reset environment
        obs = env.reset()
        done = False
        total_reward = 0
        
        print("Initial observation:", obs)
        
        while not done:
            # Get action from simulated LLM
            action = simulate_llm_response(obs)
            print("\nAction:", action)
            
            # Take step in environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            print("Reward:", reward)
            print("Observation:", obs)
            print("Info:", info)
            
        print(f"\nEpisode finished with total reward: {total_reward}")
        
    # Plot learning curve
    pipeline.plot_learning_curve("learning_curve.png")
    print("\nLearning curve saved to learning_curve.png")

if __name__ == "__main__":
    main() 