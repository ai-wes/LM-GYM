"""Meta-reasoning environment for LLM GYM."""

from typing import Dict, Any, List, Optional
import numpy as np
from ..envs.base import BaseEnv

class MetaReasoningEnv(BaseEnv):
    """Environment for training meta-reasoning capabilities."""
    
    def __init__(self, 
                 max_steps: int = 10,
                 complexity: float = 0.5,
                 required_skills: Optional[List[str]] = None):
        """Initialize meta-reasoning environment.
        
        Args:
            max_steps (int): Maximum steps per episode
            complexity (float): Task complexity (0.0 to 1.0)
            required_skills (Optional[List[str]]): Required reasoning skills
        """
        super().__init__(max_steps=max_steps)
        self.complexity = complexity
        self.required_skills = required_skills or ["planning", "evaluation"]
        
        # Valid action prefixes
        self.valid_actions = ["THINK", "ANSWER"]
        
        # Task templates
        self.task_templates = [
            {
                "type": "problem_decomposition",
                "template": "Break down the following problem into steps: {problem}",
                "skills": ["planning"]
            },
            {
                "type": "assumption_verification",
                "template": "Verify the assumptions in this reasoning: {reasoning}",
                "skills": ["evaluation"]
            },
            {
                "type": "bias_detection",
                "template": "Identify potential biases in this analysis: {analysis}",
                "skills": ["evaluation"]
            }
        ]
        
    def _generate_task(self) -> Dict[str, Any]:
        """Generate new meta-reasoning task.
        
        Returns:
            Dict[str, Any]: Task specification
        """
        # Select task template based on required skills
        valid_templates = [
            t for t in self.task_templates
            if any(skill in self.required_skills for skill in t["skills"])
        ]
        template = np.random.choice(valid_templates)
        
        # Generate task content based on complexity
        if template["type"] == "problem_decomposition":
            problem = self._generate_problem()
            content = template["template"].format(problem=problem)
        elif template["type"] == "assumption_verification":
            reasoning = self._generate_reasoning()
            content = template["template"].format(reasoning=reasoning)
        else:  # bias_detection
            analysis = self._generate_analysis()
            content = template["template"].format(analysis=analysis)
            
        return {
            "type": template["type"],
            "content": content,
            "complexity": self.complexity,
            "required_skills": template["skills"]
        }
    
    def _is_valid_action(self, action: str) -> bool:
        """Check if action is valid.
        
        Args:
            action (str): Action to validate
            
        Returns:
            bool: Whether action is valid
        """
        # Check action prefix
        action_parts = action.split(" ", 1)
        if len(action_parts) != 2:
            return False
            
        prefix, content = action_parts
        return prefix in self.valid_actions and len(content.strip()) > 0
    
    def _compute_reward(self, action: str) -> float:
        """Compute reward for action.
        
        Args:
            action (str): Action taken
            
        Returns:
            float: Reward value
        """
        action_parts = action.split(" ", 1)
        if len(action_parts) != 2:
            return -1.0
            
        prefix, content = action_parts
        
        if prefix == "THINK":
            # Reward for explicit reasoning
            return self._evaluate_thinking(content)
        elif prefix == "ANSWER":
            # Reward for final answer
            return self._evaluate_answer(content)
        
        return 0.0
    
    def _is_done(self) -> bool:
        """Check if episode is done.
        
        Returns:
            bool: Whether episode should end
        """
        # Episode ends if max steps reached or final answer given
        if self.current_step >= self.max_steps:
            return True
            
        # Check if last action was an ANSWER
        if self.history and self.history[-1].startswith("ANSWER"):
            return True
            
        return False
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation.
        
        Returns:
            Dict[str, Any]: Current observation
        """
        base_obs = super()._get_observation()
        
        # Add meta-reasoning specific information
        base_obs.update({
            "task_type": "meta_reasoning",
            "complexity": self.complexity,
            "required_skills": self.required_skills,
            "valid_actions": self.valid_actions
        })
        
        return base_obs
    
    def _generate_problem(self) -> str:
        """Generate problem for decomposition task."""
        problems = [
            "How can we reduce carbon emissions in urban areas?",
            "What are the implications of widespread AI adoption?",
            "How can we improve education accessibility?"
        ]
        return np.random.choice(problems)
    
    def _generate_reasoning(self) -> str:
        """Generate reasoning for verification task."""
        reasonings = [
            "Since it rained yesterday, the ground will be wet today.",
            "All successful people wake up early, so waking up early leads to success.",
            "Technology always makes life better, so we should adopt every new technology."
        ]
        return np.random.choice(reasonings)
    
    def _generate_analysis(self) -> str:
        """Generate analysis for bias detection task."""
        analyses = [
            "Young people are always better with technology.",
            "Traditional methods have worked for centuries, so they're the best.",
            "Modern solutions are inherently superior to old ones."
        ]
        return np.random.choice(analyses)
    
    def _evaluate_thinking(self, content: str) -> float:
        """Evaluate thinking step.
        
        Args:
            content (str): Thinking content
            
        Returns:
            float: Reward value
        """
        # Reward for length and structure
        reward = min(1.0, len(content.split()) / 50) * 0.5
        
        # Reward for using reasoning keywords
        keywords = ["because", "therefore", "however", "if", "then"]
        keyword_count = sum(1 for word in keywords if word in content.lower())
        reward += (keyword_count / len(keywords)) * 0.5
        
        return reward
    
    def _evaluate_answer(self, content: str) -> float:
        """Evaluate final answer.
        
        Args:
            content (str): Answer content
            
        Returns:
            float: Reward value
        """
        # Must have done some thinking first
        if not any(h.startswith("THINK") for h in self.history):
            return -1.0
            
        # Reward for length and structure
        reward = min(1.0, len(content.split()) / 30)
        
        # Bonus for using conclusions from thinking steps
        thinking_steps = [h.split(" ", 1)[1] for h in self.history if h.startswith("THINK")]
        for step in thinking_steps:
            if any(phrase in content.lower() for phrase in step.lower().split()):
                reward += 0.5
                
        return min(2.0, reward)  # Cap reward at 2.0 