"""Enhanced curriculum learning module for LLM GYM."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from collections import deque

@dataclass
class TaskMetrics:
    """Metrics for a completed task."""
    task_id: str
    difficulty: float
    success: bool
    reward: float
    completion_time: float
    attempts: int

class PerformanceTracker:
    """Tracks agent performance across tasks."""
    
    def __init__(self, window_size: int = 10):
        """Initialize performance tracker.
        
        Args:
            window_size (int): Size of moving average window
        """
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.task_type_stats = {}
        
    def add_metrics(self, metrics: TaskMetrics):
        """Add task metrics to history."""
        self.metrics_history.append(metrics)
        
        if metrics.task_id not in self.task_type_stats:
            self.task_type_stats[metrics.task_id] = {
                "attempts": [],
                "rewards": [],
                "completion_times": []
            }
            
        stats = self.task_type_stats[metrics.task_id]
        stats["attempts"].append(metrics.attempts)
        stats["rewards"].append(metrics.reward)
        stats["completion_times"].append(metrics.completion_time)
        
    def get_success_rate(self) -> float:
        """Get recent success rate."""
        if not self.metrics_history:
            return 0.0
        return sum(m.success for m in self.metrics_history) / len(self.metrics_history)
    
    def get_avg_reward(self) -> float:
        """Get average reward over window."""
        if not self.metrics_history:
            return 0.0
        return np.mean([m.reward for m in self.metrics_history])
    
    def get_task_stats(self, task_id: str) -> Dict[str, float]:
        """Get statistics for specific task type."""
        if task_id not in self.task_type_stats:
            return {}
            
        stats = self.task_type_stats[task_id]
        return {
            "avg_attempts": np.mean(stats["attempts"]),
            "avg_reward": np.mean(stats["rewards"]),
            "avg_completion_time": np.mean(stats["completion_times"])
        }

class AdaptiveDifficulty:
    """Manages adaptive difficulty scaling."""
    
    def __init__(self, 
                 initial_difficulty: float = 0.5,
                 min_difficulty: float = 0.1,
                 max_difficulty: float = 1.0,
                 adjustment_rate: float = 0.1):
        """Initialize adaptive difficulty.
        
        Args:
            initial_difficulty (float): Starting difficulty
            min_difficulty (float): Minimum difficulty level
            max_difficulty (float): Maximum difficulty level
            adjustment_rate (float): Rate of difficulty adjustment
        """
        self.difficulty = initial_difficulty
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.adjustment_rate = adjustment_rate
        
    def adjust(self, success_rate: float, target_rate: float = 0.7):
        """Adjust difficulty based on performance.
        
        Args:
            success_rate (float): Recent success rate
            target_rate (float): Target success rate
        """
        error = target_rate - success_rate
        adjustment = self.adjustment_rate * error
        
        self.difficulty = np.clip(
            self.difficulty + adjustment,
            self.min_difficulty,
            self.max_difficulty
        )
        
    def get_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.difficulty

class EnhancedCurriculum:
    """Enhanced curriculum with adaptive difficulty and performance tracking."""
    
    def __init__(self,
                 task_templates: Dict[str, Any],
                 window_size: int = 10,
                 initial_difficulty: float = 0.5):
        """Initialize enhanced curriculum.
        
        Args:
            task_templates (Dict[str, Any]): Task generation templates
            window_size (int): Performance tracking window size
            initial_difficulty (float): Starting difficulty level
        """
        self.task_templates = task_templates
        self.tracker = PerformanceTracker(window_size)
        self.difficulty = AdaptiveDifficulty(initial_difficulty)
        self.current_task = None
        
    def generate_task(self) -> Dict[str, Any]:
        """Generate next task based on current difficulty."""
        difficulty = self.difficulty.get_difficulty()
        
        # Select task type based on performance history
        task_type = self._select_task_type()
        template = self.task_templates[task_type]
        
        # Scale parameters based on difficulty
        params = self._scale_parameters(template["parameters"], difficulty)
        
        self.current_task = {
            "id": task_type,
            "difficulty": difficulty,
            "parameters": params,
            "start_time": None
        }
        
        return self.current_task
    
    def _select_task_type(self) -> str:
        """Select next task type based on performance."""
        if not self.task_templates:
            raise ValueError("No task templates available")
            
        # Simple random selection for now
        # TODO: Implement more sophisticated selection based on learning curves
        return np.random.choice(list(self.task_templates.keys()))
    
    def _scale_parameters(self, 
                         base_params: Dict[str, Any], 
                         difficulty: float) -> Dict[str, Any]:
        """Scale task parameters based on difficulty.
        
        Args:
            base_params (Dict[str, Any]): Base parameters from template
            difficulty (float): Current difficulty level
            
        Returns:
            Dict[str, Any]: Scaled parameters
        """
        scaled = {}
        for key, value in base_params.items():
            if isinstance(value, (int, float)):
                # Scale numeric parameters
                scaled[key] = value * difficulty
            elif isinstance(value, list):
                # Adjust list length based on difficulty
                length = max(1, int(len(value) * difficulty))
                scaled[key] = value[:length]
            else:
                # Keep non-numeric parameters as is
                scaled[key] = value
        return scaled
    
    def complete_task(self, 
                     success: bool, 
                     reward: float,
                     attempts: int) -> None:
        """Record task completion metrics.
        
        Args:
            success (bool): Whether task was completed successfully
            reward (float): Reward received
            attempts (int): Number of attempts made
        """
        if not self.current_task:
            return
            
        metrics = TaskMetrics(
            task_id=self.current_task["id"],
            difficulty=self.current_task["difficulty"],
            success=success,
            reward=reward,
            completion_time=0.0,  # TODO: Track actual completion time
            attempts=attempts
        )
        
        self.tracker.add_metrics(metrics)
        success_rate = self.tracker.get_success_rate()
        self.difficulty.adjust(success_rate)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        return {
            "current_difficulty": self.difficulty.get_difficulty(),
            "success_rate": self.tracker.get_success_rate(),
            "avg_reward": self.tracker.get_avg_reward(),
            "task_stats": {
                task_id: self.tracker.get_task_stats(task_id)
                for task_id in self.task_templates.keys()
            }
        } 