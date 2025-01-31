from typing import Dict, List, Any, Optional, Callable
import numpy as np
from dataclasses import dataclass

@dataclass
class RewardComponent:
    """A component of the reward signal."""
    name: str
    weight: float
    compute_fn: Callable[[Dict[str, Any]], float]
    
class RewardManager:
    """Manages and combines multiple reward signals."""
    
    def __init__(self):
        self.components: List[RewardComponent] = []
        self._setup_default_components()
        
    def _setup_default_components(self):
        """Set up default reward components."""
        # Task completion reward
        self.add_component(
            "completion",
            1.0,
            lambda state: float(state.get("task_completed", False))
        )
        
        # Step-based penalties to encourage efficiency
        self.add_component(
            "step_penalty",
            -0.01,
            lambda state: 1.0  # Applied each step
        )
        
        # Chain-of-thought quality
        self.add_component(
            "reasoning_quality",
            0.5,
            self._evaluate_reasoning_quality
        )
        
    def add_component(self, name: str, weight: float, 
                     compute_fn: Callable[[Dict[str, Any]], float]):
        """Add a new reward component.
        
        Args:
            name (str): Name of the component
            weight (float): Weight in final reward
            compute_fn (Callable): Function to compute this component's reward
        """
        component = RewardComponent(name=name, weight=weight, compute_fn=compute_fn)
        self.components.append(component)
        
    def compute_reward(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Compute total reward and individual components.
        
        Args:
            state (Dict[str, Any]): Current state information
            
        Returns:
            Dict[str, float]: Total reward and component breakdown
        """
        rewards = {}
        total_reward = 0.0
        
        for component in self.components:
            try:
                component_reward = component.compute_fn(state) * component.weight
                rewards[component.name] = component_reward
                total_reward += component_reward
            except Exception as e:
                print(f"Error computing reward for {component.name}: {e}")
                rewards[component.name] = 0.0
                
        rewards["total"] = total_reward
        return rewards
    
    def _evaluate_reasoning_quality(self, state: Dict[str, Any]) -> float:
        """Evaluate the quality of chain-of-thought reasoning.
        
        Args:
            state (Dict[str, Any]): Current state with reasoning information
            
        Returns:
            float: Quality score (0.0 to 1.0)
        """
        reasoning = state.get("chain_of_thought", "")
        if not reasoning:
            return 0.0
            
        score = 0.0
        words = reasoning.split()
        
        # Length-based component (up to 0.3)
        length_score = min(len(words) / 50.0, 1.0) * 0.3
        score += length_score
        
        # Structure-based component (up to 0.3)
        has_therefore = "therefore" in reasoning.lower() or "thus" in reasoning.lower()
        has_because = "because" in reasoning.lower()
        has_if = "if" in reasoning.lower()
        structure_score = sum([has_therefore, has_because, has_if]) * 0.1
        score += structure_score
        
        # Coherence-based component (up to 0.4)
        # This is a simple heuristic; in practice, you might want to use
        # more sophisticated NLP techniques
        sentences = reasoning.split('.')
        if len(sentences) > 1:
            score += 0.4
            
        return score
        
class HumanFeedbackRewardComponent(RewardComponent):
    """Reward component that incorporates human feedback."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(
            name="human_feedback",
            weight=weight,
            compute_fn=self._compute_human_feedback
        )
        self.feedback_history: List[float] = []
        
    def _compute_human_feedback(self, state: Dict[str, Any]) -> float:
        """Compute reward based on human feedback.
        
        Args:
            state (Dict[str, Any]): Current state
            
        Returns:
            float: Reward value
        """
        feedback = state.get("human_feedback", None)
        if feedback is not None:
            self.feedback_history.append(feedback)
            return feedback
        return 0.0
        
    def get_feedback_stats(self) -> Dict[str, float]:
        """Get statistics about collected human feedback.
        
        Returns:
            Dict[str, float]: Statistics about feedback
        """
        if not self.feedback_history:
            return {"mean": 0.0, "std": 0.0, "count": 0}
            
        return {
            "mean": np.mean(self.feedback_history),
            "std": np.std(self.feedback_history),
            "count": len(self.feedback_history)
        }
        
class CurriculumRewardComponent(RewardComponent):
    """Reward component that scales with task difficulty."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(
            name="curriculum_scaling",
            weight=weight,
            compute_fn=self._compute_curriculum_reward
        )
        
    def _compute_curriculum_reward(self, state: Dict[str, Any]) -> float:
        """Compute reward scaled by task difficulty.
        
        Args:
            state (Dict[str, Any]): Current state
            
        Returns:
            float: Scaled reward value
        """
        base_reward = float(state.get("task_completed", False))
        difficulty = state.get("task_difficulty", 0.5)
        
        # Scale reward by difficulty (harder tasks = higher reward)
        return base_reward * (1.0 + difficulty) 