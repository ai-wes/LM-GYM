from typing import Dict, Any, Tuple, Optional, List
from ..envs.base import BaseEnv
from ..curriculum.generator import CurriculumGenerator, TaskConfig
from ..utils.rewards import RewardManager, HumanFeedbackRewardComponent

class DialogueEnv(BaseEnv):
    """
    Environment for dialogue-based tasks.
    The agent must engage in conversations with specific goals and constraints.
    """
    
    def __init__(self, 
                 use_curriculum: bool = True,
                 use_human_feedback: bool = False,
                 max_turns: int = 10):
        """Initialize dialogue environment.
        
        Args:
            use_curriculum (bool): Whether to use curriculum learning
            use_human_feedback (bool): Whether to incorporate human feedback
            max_turns (int): Maximum conversation turns
        """
        super().__init__()
        self.max_steps = max_turns
        self.use_curriculum = use_curriculum
        self.use_human_feedback = use_human_feedback
        
        # Initialize components
        self.curriculum = CurriculumGenerator() if use_curriculum else None
        self.reward_manager = RewardManager()
        
        if use_human_feedback:
            self.reward_manager.add_component(
                "human_feedback",
                1.0,
                HumanFeedbackRewardComponent()
            )
            
        # Current episode state
        self.current_task = None
        self.conversation_history = []
        self.required_elements = []
        self.goals_achieved = set()
        
    def reset(self) -> Dict[str, Any]:
        """Reset the environment for a new episode."""
        super().reset()
        
        # Generate new task
        if self.use_curriculum:
            self.current_task = self.curriculum.generate_task("dialogue")
        else:
            self.current_task = self.curriculum.generate_task("dialogue")
            
        # Setup conversation
        self.conversation_history = []
        self.required_elements = self.current_task.parameters["required_elements"]
        self.goals_achieved = set()
        
        # Add initial context to conversation
        scenario = self.current_task.parameters["scenario"]
        self._add_system_message(f"Scenario: {scenario}")
        
        return self._get_observation()
        
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        if self.done:
            return self._get_observation(), 0.0, True, {"info": "Episode already finished"}
            
        # Add agent's message to conversation
        self.conversation_history.append({"role": "assistant", "content": action})
        
        # Evaluate response
        reward_info = self._evaluate_response(action)
        
        # Generate system response (in practice, this would be more sophisticated)
        if not self.done:
            self._generate_system_response()
            
        # Check if conversation should end
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True
            
        # Check if all required elements are present
        if all(goal in self.goals_achieved for goal in self.required_elements):
            self.done = True
            
        return self._get_observation(), reward_info["total"], self.done, reward_info
        
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        obs = {
            "conversation": self.conversation_history,
            "required_elements": self.required_elements,
            "goals_achieved": list(self.goals_achieved),
            "turns_remaining": self.max_steps - self.current_step
        }
        
        if self.use_curriculum:
            obs["current_difficulty"] = self.curriculum.current_difficulty
            
        return obs
        
    def _add_system_message(self, content: str):
        """Add a system message to the conversation."""
        self.conversation_history.append({"role": "system", "content": content})
        
    def _evaluate_response(self, response: str) -> Dict[str, float]:
        """Evaluate agent's response and update goals."""
        # Check for required elements in response
        for element in self.required_elements:
            if element not in self.goals_achieved:
                if self._check_element(element, response):
                    self.goals_achieved.add(element)
                    
        # Compute reward
        state = {
            "task_completed": len(self.goals_achieved) == len(self.required_elements),
            "task_difficulty": self.current_task.difficulty if self.current_task else 0.5,
            "goals_achieved": len(self.goals_achieved),
            "total_goals": len(self.required_elements),
            "response": response
        }
        
        return self.reward_manager.compute_reward(state)
        
    def _check_element(self, element: str, response: str) -> bool:
        """Check if a required element is present in the response."""
        response = response.lower()
        
        if element == "politeness":
            polite_words = {"please", "thank", "appreciate", "would you", "could you"}
            return any(word in response for word in polite_words)
            
        elif element == "coherence":
            # Simple check for minimum length and sentence structure
            return len(response.split()) >= 5 and "." in response
            
        elif element == "emotional_awareness":
            emotion_words = {"feel", "emotion", "happy", "sad", "understand", "empathize"}
            return any(word in response for word in emotion_words)
            
        elif element == "conflict_resolution":
            resolution_words = {"resolve", "solution", "agree", "compromise", "suggest"}
            return any(word in response for word in resolution_words)
            
        elif element == "cultural_sensitivity":
            culture_words = {"culture", "background", "tradition", "respect", "diverse"}
            return any(word in response for word in culture_words)
            
        return False
        
    def _generate_system_response(self):
        """Generate a response from the system."""
        scenario = self.current_task.parameters["scenario"]
        
        if scenario == "simple_question_answer":
            response = "Could you explain that in more detail?"
        elif scenario == "negotiation":
            response = "That's interesting. What else can you offer?"
        elif scenario == "debate":
            response = "I see your point, but have you considered the opposite view?"
        elif scenario == "emotional_support":
            response = "Thank you for your support. How else can you help?"
        else:  # technical_support
            response = "I'm still having the issue. What should I try next?"
            
        self._add_system_message(response)

    def _compute_reward(self, 
                       state: Dict[str, Any], 
                       action: str, 
                       result: Any) -> float:
        """Compute reward for dialogue actions.
        
        Args:
            state: Current dialogue state
            action: Agent's response
            result: Response evaluation results
        
        Returns:
            float: Computed reward
        """
        base_reward = 0.0
        
        # Check for required elements
        for element in self.required_elements:
            if element not in self.goals_achieved and self._check_element(action, element):
                self.goals_achieved.add(element)
                base_reward += 0.3
        
        # Evaluate response quality
        if self.use_human_feedback:
            human_feedback = self.reward_manager.get_component("human_feedback").evaluate(action)
            base_reward += human_feedback
        
        # Penalize repetition
        if any(turn["content"] == action for turn in self.conversation_history[:-1]):
            base_reward -= 0.2
        
        # Scale by task difficulty
        difficulty = self.current_task.difficulty if self.current_task else 0.5
        return base_reward * (1.0 + difficulty)

    def _generate_task(self) -> TaskConfig:
        """Generate a new dialogue task.
        
        Returns:
            TaskConfig: New task configuration
        """
        difficulty = self.curriculum.current_difficulty if self.use_curriculum else 0.5
        
        # Scale requirements with difficulty
        num_required = max(1, int(2 + difficulty * 3))
        
        scenarios = [
            "customer_service",
            "negotiation",
            "casual_chat",
            "technical_support",
            "debate"
        ]
        
        return TaskConfig(
            difficulty=difficulty,
            task_type="dialogue",
            parameters={
                "scenario": scenarios[int(difficulty * (len(scenarios)-1))],
                "required_elements": self._get_required_elements(num_required),
                "max_turns": self.max_steps
            }
        )

    def _is_valid_action(self, action: str) -> bool:
        """Check if the dialogue action is valid.
        
        Args:
            action: Agent's response
        
        Returns:
            bool: Whether action is valid
        """
        # Basic validation
        if not isinstance(action, str) or len(action.strip()) == 0:
            return False
        
        # Check for reasonable length
        if len(action.split()) > 100:  # Arbitrary max length
            return False
        
        # Check for basic dialogue structure
        has_greeting = any(word in action.lower() for word in ["hello", "hi", "hey", "greetings"])
        has_content = len(action.split()) > 1
        
        return has_greeting or has_content

    def _check_element(self, action: str, element: str) -> bool:
        """Check if an action contains a required dialogue element.
        
        Args:
            action: Agent's response
            element: Required element to check for
        
        Returns:
            bool: Whether element is present
        """
        element_patterns = {
            "greeting": ["hello", "hi", "hey", "greetings"],
            "question": ["?", "what", "how", "when", "where", "why"],
            "acknowledgment": ["understand", "got it", "i see", "okay"],
            "clarification": ["mean", "could you explain", "to clarify"],
            "conclusion": ["thank", "goodbye", "bye", "farewell"]
        }
        
        if element in element_patterns:
            return any(pattern in action.lower() for pattern in element_patterns[element])
        return False

    def _get_required_elements(self, num_required: int) -> List[str]:
        """Get a list of required dialogue elements.
        
        Args:
            num_required: Number of elements to require
        
        Returns:
            List[str]: Required elements
        """
        all_elements = ["greeting", "question", "acknowledgment", "clarification", "conclusion"]
        return all_elements[:num_required] 