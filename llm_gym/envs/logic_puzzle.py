"""Logic puzzle environment for LLM GYM."""

from typing import Dict, Any, Optional, List
import numpy as np
from .base import BaseEnv

class LogicPuzzleEnv(BaseEnv):
    """Environment for logic puzzle solving."""
    
    def __init__(self, 
                 max_steps: int = 10,
                 complexity: float = 0.5,
                 required_steps: Optional[int] = None):
        """Initialize logic puzzle environment.
        
        Args:
            max_steps (int): Maximum steps per episode
            complexity (float): Task complexity (0.0 to 1.0)
            required_steps (Optional[int]): Required solution steps
        """
        super().__init__(max_steps=max_steps)
        self.complexity = complexity
        self.required_steps = required_steps or 3
        
        # Valid action prefixes
        self.valid_actions = ["SOLVE", "QUERY"]
        
        # Current episode state
        self.current_premises = []
        self.current_solution = None
        self.attempted_solutions = []
        
        # Track performance metrics
        self.total_correct = 0
        self.total_attempts = 0
        
    def _generate_task(self) -> Dict[str, Any]:
        """Generate new logic puzzle task.
        
        Returns:
            Dict[str, Any]: Task specification
        """
        # Scale complexity with difficulty
        num_premises = max(2, int(3 + self.complexity * 3))
        num_variables = max(2, int(2 + self.complexity * 4))
        
        # Generate premises and solution
        self.current_premises = []
        variables = [chr(65 + i) for i in range(num_variables)]  # A, B, C, etc.
        
        relation_templates = [
            "If {0} then {1}",
            "{0} implies {1}",
            "When {0} occurs, {1} follows",
            "{0} leads to {1}",
            "{0} causes {1}"
        ]
        
        # Generate premises
        for i in range(num_premises):
            template = relation_templates[i % len(relation_templates)]
            var1 = variables[i % num_variables]
            var2 = variables[(i + 1) % num_variables]
            premise = template.format(var1, var2)
            self.current_premises.append(premise)
            
        # Generate solution
        self.current_solution = f"Therefore, {variables[-1]}"
        
        return {
            "type": "logic_puzzle",
            "content": "\n".join(self.current_premises),
            "complexity": self.complexity,
            "required_steps": self.required_steps
        }
    
    def _is_valid_action(self, action: str) -> bool:
        """Check if action is valid.
        
        Args:
            action (str): Action to validate
            
        Returns:
            bool: Whether action is valid
        """
        try:
            command, content = action.split(" ", 1)
            command = command.upper()
            return command in self.valid_actions and len(content.strip()) > 0
        except ValueError:
            return False
    
    def _compute_reward(self, action: str) -> float:
        """Compute reward for action.
        
        Args:
            action (str): Action taken
            
        Returns:
            float: Reward value
        """
        try:
            command, content = action.split(" ", 1)
            command = command.upper()
        except ValueError:
            return -1.0
            
        if command == "SOLVE":
            # Track solution attempt
            self.attempted_solutions.append(content)
            
            # Check if solution is correct
            is_correct = self.current_solution.lower() in content.lower()
            
            if is_correct:
                self.total_correct += 1
                # Reward for correct solution with attempt penalty
                base_reward = 1.0
                attempts_penalty = 0.2 * (len(self.attempted_solutions) - 1)
                return max(0.2, base_reward - attempts_penalty)
            else:
                return -0.2  # Penalty for incorrect solution
                
        elif command == "QUERY":
            # Check query relevance
            relevant_premises = [
                premise for premise in self.current_premises
                if any(word in content.lower() for word in premise.lower().split())
            ]
            
            if len(relevant_premises) > 0:
                # Reward for relevant query
                base_reward = 0.1
                if len(self.attempted_solutions) == 0:
                    base_reward += 0.1  # Bonus for querying before solving
                return base_reward
            else:
                return -0.1  # Penalty for irrelevant query
                
        return 0.0
    
    def _is_done(self) -> bool:
        """Check if episode is done.
        
        Returns:
            bool: Whether episode should end
        """
        # Episode ends if max steps reached or correct solution found
        if self.current_step >= self.max_steps:
            return True
            
        # Check if last action was a correct SOLVE
        if self.history:
            last_action = self.history[-1]
            if last_action.startswith("SOLVE"):
                _, content = last_action.split(" ", 1)
                if self.current_solution.lower() in content.lower():
                    return True
                    
        return False
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation.
        
        Returns:
            Dict[str, Any]: Current observation
        """
        base_obs = super()._get_observation()
        
        # Add logic puzzle specific information
        base_obs.update({
            "task_type": "logic_puzzle",
            "premises": self.current_premises,
            "num_attempts": len(self.attempted_solutions),
            "complexity": self.complexity,
            "valid_actions": self.valid_actions
        })
        
        return base_obs 