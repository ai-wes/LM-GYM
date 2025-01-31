"""Base environment class for LLM GYM."""

from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np

class BaseEnv(ABC):
    """Abstract base class for all LLM GYM environments."""
    
    def __init__(self, max_steps: int = 100):
        """Initialize base environment.
        
        Args:
            max_steps (int): Maximum steps per episode
        """
        self.max_steps = max_steps
        self.current_step = 0
        self.episode_count = 0
        self.history = []
        self.current_task = None
        self.done = False
        self.metadata = {}
        self.reward_range = (-float('inf'), float('inf'))
        
    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state.
        
        Returns:
            Dict[str, Any]: Initial observation
        """
        self.current_step = 0
        self.episode_count += 1
        self.history = []
        self.done = False
        
        # Generate new task
        self.current_task = self._generate_task()
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take step in environment.
        
        Args:
            action (str): Action to take
            
        Returns:
            Tuple containing:
            - Dict[str, Any]: New observation
            - float: Reward
            - bool: Whether episode is done
            - Dict[str, Any]: Additional info
        """
        if self.done:
            raise RuntimeError("Episode is done, call reset() first")
            
        # Validate action
        if not self._is_valid_action(action):
            return self._get_observation(), -1.0, True, {"error": "Invalid action"}
        
        # Process action
        self.history.append(action)
        reward = self._compute_reward(action)
        
        # Update state
        self.current_step += 1
        self.done = self._is_done()
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, self.done, info
    
    @abstractmethod
    def _generate_task(self) -> Dict[str, Any]:
        """Generate new task for episode.
        
        Returns:
            Dict[str, Any]: Task specification
        """
        pass
    
    @abstractmethod
    def _is_valid_action(self, action: str) -> bool:
        """Check if action is valid.
        
        Args:
            action (str): Action to validate
            
        Returns:
            bool: Whether action is valid
        """
        pass
    
    @abstractmethod
    def _compute_reward(self, action: str) -> float:
        """Compute reward for action.
        
        Args:
            action (str): Action taken
            
        Returns:
            float: Reward value
        """
        pass
    
    def _is_done(self) -> bool:
        """Check if episode is done.
        
        Returns:
            bool: Whether episode should end
        """
        # Episode ends if max steps reached
        if self.current_step >= self.max_steps:
            return True
            
        return False
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation.
        
        Returns:
            Dict[str, Any]: Current observation
        """
        return {
            "task": self.current_task,
            "step": self.current_step,
            "max_steps": self.max_steps,
            "history": self.history.copy()
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state.
        
        Returns:
            Dict[str, Any]: Additional information
        """
        return {
            "episode": self.episode_count,
            "steps_taken": self.current_step
        }
    
    def close(self):
        """Clean up environment resources."""
        pass
        
    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility.
        
        Args:
            seed (Optional[int]): Random seed
        """
        np.random.seed(seed)
        
    @property
    def unwrapped(self):
        """Get the base environment.
        
        Returns:
            The base environment
        """
        return self 