"""Distributed training support for LLM GYM."""

import ray
from typing import List, Dict, Any, Tuple
from ..envs.base import BaseEnv

@ray.remote
class RemoteEnvironment:
    """Wrapper for environment parallelization using Ray."""
    
    def __init__(self, env_class: type, config: Dict[str, Any]):
        self.env = env_class(**config)
        self.episode_history = []
        
    def reset(self) -> Dict[str, Any]:
        """Reset the environment."""
        return self.env.reset()
        
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        return self.env.step(action)
        
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get episode history metrics."""
        return self.episode_history
        
    def run_episodes(self, num_episodes: int) -> List[Dict[str, Any]]:
        """Run multiple episodes and collect results."""
        results = []
        for _ in range(num_episodes):
            episode_result = self._run_single_episode()
            results.append(episode_result)
            self.episode_history.append(episode_result)
        return results
        
    def _run_single_episode(self) -> Dict[str, Any]:
        """Run a single episode and return results."""
        obs = self.env.reset()
        done = False
        total_reward = 0
        steps = []
        
        while not done:
            # In practice, you would have an agent here
            action = "dummy_action"  # Replace with actual agent
            next_obs, reward, done, info = self.env.step(action)
            total_reward += reward
            steps.append({
                "observation": obs,
                "action": action,
                "reward": reward,
                "next_observation": next_obs,
                "done": done,
                "info": info
            })
            obs = next_obs
            
        return {
            "total_reward": total_reward,
            "num_steps": len(steps),
            "steps": steps
        }

class DistributedTrainer:
    """Manages distributed training across multiple environments."""
    
    def __init__(self, num_workers: int, env_class: type, env_config: Dict[str, Any]):
        """Initialize distributed trainer.
        
        Args:
            num_workers (int): Number of parallel workers
            env_class (type): Environment class to use
            env_config (Dict[str, Any]): Environment configuration
        """
        ray.init(ignore_reinit_error=True)
        self.workers = [RemoteEnvironment.remote(env_class, env_config) 
                       for _ in range(num_workers)]
        
    def parallel_rollout(self, num_episodes: int) -> List[Dict[str, Any]]:
        """Execute episodes in parallel across workers.
        
        Args:
            num_episodes (int): Total number of episodes to run
            
        Returns:
            List[Dict[str, Any]]: Results from all episodes
        """
        results = []
        episodes_per_worker = num_episodes // len(self.workers)
        remaining_episodes = num_episodes % len(self.workers)
        
        # Distribute episodes among workers
        futures = []
        for i, worker in enumerate(self.workers):
            worker_episodes = episodes_per_worker
            if i < remaining_episodes:
                worker_episodes += 1
            futures.append(
                worker.run_episodes.remote(worker_episodes)
            )
            
        # Collect results
        worker_results = ray.get(futures)
        for result in worker_results:
            results.extend(result)
            
        return results
        
    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Get metrics from all workers."""
        futures = [worker.get_metrics.remote() for worker in self.workers]
        return ray.get(futures)
        
    def shutdown(self):
        """Shutdown Ray."""
        ray.shutdown() 