"""Adaptive reward management module for LLM GYM."""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class RewardComponent:
    """Component of the reward signal."""
    name: str
    weight: float
    min_value: float
    max_value: float
    threshold: float
    is_binary: bool = False

class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    @abstractmethod
    def compute(self, state: Dict[str, Any], action: str) -> float:
        """Compute reward value.
        
        Args:
            state (Dict[str, Any]): Current environment state
            action (str): Agent's action
            
        Returns:
            float: Computed reward value
        """
        pass

class TaskCompletionReward(RewardFunction):
    """Reward for task completion."""
    
    def compute(self, state: Dict[str, Any], action: str) -> float:
        """Compute task completion reward."""
        return 1.0 if state.get("task_completed", False) else 0.0

class EfficiencyReward(RewardFunction):
    """Reward for efficient solution."""
    
    def compute(self, state: Dict[str, Any], action: str) -> float:
        """Compute efficiency reward."""
        steps_taken = state.get("steps_taken", 0)
        max_steps = state.get("max_steps", 100)
        return 1.0 - (steps_taken / max_steps)

class SafetyReward(RewardFunction):
    """Reward for safe behavior."""
    
    def compute(self, state: Dict[str, Any], action: str) -> float:
        """Compute safety reward."""
        violations = state.get("safety_violations", 0)
        return -1.0 if violations > 0 else 0.0

class NoveltyReward(RewardFunction):
    """Reward for novel solutions."""
    
    def __init__(self, memory_size: int = 100):
        """Initialize novelty reward.
        
        Args:
            memory_size (int): Size of solution memory
        """
        self.memory_size = memory_size
        self.solution_memory = []
        
    def compute(self, state: Dict[str, Any], action: str) -> float:
        """Compute novelty reward."""
        if not state.get("task_completed", False):
            return 0.0
            
        solution = state.get("solution", "")
        if not solution:
            return 0.0
            
        # Check similarity with previous solutions
        if solution in self.solution_memory:
            return 0.0
            
        # Add to memory
        self.solution_memory.append(solution)
        if len(self.solution_memory) > self.memory_size:
            self.solution_memory.pop(0)
            
        return 1.0

class AdaptiveRewardManager:
    """Manages adaptive reward computation with multiple objectives."""
    
    def __init__(self,
                 components: List[RewardComponent],
                 learning_rate: float = 0.01,
                 temperature: float = 1.0):
        """Initialize reward manager.
        
        Args:
            components (List[RewardComponent]): Reward components
            learning_rate (float): Learning rate for weight updates
            temperature (float): Temperature for weight smoothing
        """
        self.components = {c.name: c for c in components}
        self.learning_rate = learning_rate
        self.temperature = temperature
        
        # Initialize reward functions
        self.reward_functions = {
            "task_completion": TaskCompletionReward(),
            "efficiency": EfficiencyReward(),
            "safety": SafetyReward(),
            "novelty": NoveltyReward()
        }
        
        # Performance history
        self.performance_history = []
        
    def compute_reward(self, 
                      state: Dict[str, Any],
                      action: str) -> Tuple[float, Dict[str, float]]:
        """Compute weighted reward from all components.
        
        Args:
            state (Dict[str, Any]): Current environment state
            action (str): Agent's action
            
        Returns:
            Tuple[float, Dict[str, float]]: Total reward and component values
        """
        component_values = {}
        
        for name, component in self.components.items():
            if name not in self.reward_functions:
                continue
                
            # Compute raw value
            value = self.reward_functions[name].compute(state, action)
            
            # Apply normalization
            normalized = self._normalize_value(
                value, component.min_value, component.max_value
            )
            
            # Apply thresholding if binary
            if component.is_binary:
                normalized = 1.0 if normalized >= component.threshold else 0.0
                
            component_values[name] = normalized
            
        # Compute weighted sum
        total_reward = sum(
            self.components[name].weight * value
            for name, value in component_values.items()
        )
        
        return total_reward, component_values
    
    def update_weights(self, 
                      performance_metrics: Dict[str, float],
                      target_metrics: Dict[str, float]):
        """Update component weights based on performance.
        
        Args:
            performance_metrics (Dict[str, float]): Current performance
            target_metrics (Dict[str, float]): Target performance
        """
        self.performance_history.append(performance_metrics)
        
        # Compute errors
        errors = {
            name: target_metrics.get(name, 0.0) - performance_metrics.get(name, 0.0)
            for name in self.components
        }
        
        # Update weights
        for name, component in self.components.items():
            if name not in errors:
                continue
                
            error = errors[name]
            update = self.learning_rate * error
            
            # Apply temperature scaling
            update = update * self.temperature
            
            # Update weight with clipping
            new_weight = np.clip(
                component.weight + update,
                0.0,  # Minimum weight
                1.0   # Maximum weight
            )
            
            self.components[name] = RewardComponent(
                name=component.name,
                weight=new_weight,
                min_value=component.min_value,
                max_value=component.max_value,
                threshold=component.threshold,
                is_binary=component.is_binary
            )
            
        # Normalize weights
        self._normalize_weights()
        
    def _normalize_value(self, 
                        value: float,
                        min_value: float,
                        max_value: float) -> float:
        """Normalize value to [0, 1] range.
        
        Args:
            value (float): Raw value
            min_value (float): Minimum value
            max_value (float): Maximum value
            
        Returns:
            float: Normalized value
        """
        if max_value == min_value:
            return 0.0
        return (value - min_value) / (max_value - min_value)
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1."""
        total = sum(c.weight for c in self.components.values())
        if total == 0:
            return
            
        for name, component in self.components.items():
            normalized_weight = component.weight / total
            self.components[name] = RewardComponent(
                name=component.name,
                weight=normalized_weight,
                min_value=component.min_value,
                max_value=component.max_value,
                threshold=component.threshold,
                is_binary=component.is_binary
            )
            
    def get_component_weights(self) -> Dict[str, float]:
        """Get current component weights.
        
        Returns:
            Dict[str, float]: Component weights
        """
        return {name: c.weight for name, c in self.components.items()}
    
    def get_performance_history(self) -> List[Dict[str, float]]:
        """Get performance history.
        
        Returns:
            List[Dict[str, float]]: Performance metrics history
        """
        return self.performance_history 