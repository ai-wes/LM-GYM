"""
Demo script showing how to use multiple LLM GYM environments together.
This simulates an LLM agent learning different cognitive strategies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_gym.envs import (
    MetaReasoningEnv,
    LogicPuzzleEnv,
    DialogueEnv,
    MemoryBufferEnv
)
from llm_gym.evaluation import EvaluationPipeline

def simulate_meta_reasoning(observation):
    """Simulate meta-reasoning responses."""
    task = observation.get("task", {})
    
    if not observation.get("chain_of_thought"):
        # First step - provide reasoning
        reasoning = "Let me analyze this step by step:\n"
        for r_type in task.get("reflection_types", []):
            reasoning += f"1. For {r_type}, I need to consider...\n"
        return f"THINK {reasoning}"
    else:
        # Provide final answer
        answer = "Based on my analysis:\n"
        for r_type in task.get("reflection_types", []):
            answer += f"Regarding {r_type}, I conclude that...\n"
        return f"ANSWER {answer}"

def simulate_logic_puzzle(observation):
    """Simulate logic puzzle responses."""
    premises = observation.get("premises", [])
    
    if observation.get("num_attempts", 0) == 0:
        # First try to solve
        return f"SOLVE Based on the premises {', '.join(premises)}, I conclude..."
    else:
        # Try querying about relationships
        return "QUERY How are the first two premises related?"

def simulate_dialogue(observation):
    """Simulate dialogue responses."""
    conversation = observation.get("conversation", [])
    required = observation.get("required_elements", [])
    
    # Get last message
    last_message = conversation[-1]["content"] if conversation else ""
    
    # Generate response with required elements
    response = "I understand. "
    if "politeness" in required:
        response += "Please let me help you. "
    if "emotional_awareness" in required:
        response += "I can see how you feel. "
    if "conflict_resolution" in required:
        response += "Let's find a solution together. "
        
    return response

def simulate_memory(observation):
    """Simulate memory responses."""
    query = observation.get("current_query", "")
    retrieved = observation.get("retrieved_items", [])
    
    if "store" in query.lower():
        # Store new information
        return "STORE The capital of France is Paris"
    elif retrieved:
        # Answer based on retrieved items
        return f"ANSWER Based on my memory: {retrieved[0]['content']}"
    else:
        # Try to retrieve relevant information
        return "RETRIEVE capital"

def run_environment(env_class, simulator, num_episodes=3):
    """Run episodes in an environment."""
    env = env_class(use_curriculum=True)
    pipeline = EvaluationPipeline(env)
    
    print(f"\nRunning {env_class.__name__}")
    print("=" * 50)
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}")
        print("-" * 30)
        
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Get action from simulator
            action = simulator(obs)
            print(f"\nAction: {action}")
            
            # Take step
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"Reward: {reward}")
            print(f"Info: {info}")
            
        print(f"\nEpisode finished with total reward: {total_reward}")
        
    # Plot learning curve
    pipeline.plot_learning_curve(f"{env_class.__name__}_learning.png")
    return pipeline

def main():
    # Dictionary of environments and their simulators
    environments = {
        MetaReasoningEnv: simulate_meta_reasoning,
        LogicPuzzleEnv: simulate_logic_puzzle,
        DialogueEnv: simulate_dialogue,
        MemoryBufferEnv: simulate_memory
    }
    
    # Run all environments
    results = {}
    for env_class, simulator in environments.items():
        pipeline = run_environment(env_class, simulator)
        results[env_class.__name__] = pipeline.metrics
        
    # Print summary
    print("\nOverall Results")
    print("=" * 50)
    for env_name, metrics in results.items():
        if metrics:
            latest = metrics[-1]
            print(f"\n{env_name}:")
            print(f"Mean Reward: {latest.get('mean_reward', 0):.2f}")
            print(f"Mean Episode Length: {latest.get('mean_episode_length', 0):.2f}")
            if 'mean_chain_length' in latest:
                print(f"Mean Chain Length: {latest['mean_chain_length']:.2f}")

if __name__ == "__main__":
    main() 