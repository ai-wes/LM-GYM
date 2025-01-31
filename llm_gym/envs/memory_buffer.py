from typing import Dict, Any, Tuple, Optional, List
from .base import BaseEnv
from ..curriculum.generator import CurriculumGenerator, TaskConfig
from ..utils.rewards import RewardManager
import random

class MemoryBufferEnv(BaseEnv):
    """
    Environment for memory and continual learning tasks.
    The agent must store and retrieve information across episodes.
    """
    
    def __init__(self, 
                 use_curriculum: bool = True,
                 max_steps: int = 10,
                 buffer_size: int = 100):
        """Initialize memory buffer environment.
        
        Args:
            use_curriculum (bool): Whether to use curriculum learning
            max_steps (int): Maximum steps per episode
            buffer_size (int): Maximum size of memory buffer
        """
        super().__init__()
        self.max_steps = max_steps
        self.use_curriculum = use_curriculum
        self.buffer_size = buffer_size
        
        # Initialize components
        self.curriculum = CurriculumGenerator() if use_curriculum else None
        self.reward_manager = RewardManager()
        
        # Memory state
        self.memory_buffer = []
        self.current_query = None
        self.target_response = None
        self.retrieved_items = []
        
        # Performance tracking
        self.successful_retrievals = 0
        self.total_retrievals = 0
        
    def reset(self) -> Dict[str, Any]:
        """Reset the environment for a new episode."""
        super().reset()
        
        # Generate new task
        self.current_task = self._generate_task()
        self.current_query = None
        self.retrieved_items = []
        
        return self._get_observation()
        
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        if self.done:
            return self._get_observation(), 0.0, True, {"info": "Episode already finished"}
            
        # Parse action
        try:
            command, content = action.split(" ", 1)
            command = command.upper()
        except ValueError:
            return self._get_observation(), -1.0, False, {"error": "Invalid action format"}
            
        # Process action
        if command == "STORE":
            # Store information in memory
            reward_info = self._store_memory(content)
            
        elif command == "RETRIEVE":
            # Retrieve information from memory
            reward_info = self._retrieve_memory(content)
            
        elif command == "ANSWER":
            # Answer the current query
            is_correct = self._evaluate_answer(content)
            reward_info = self._compute_answer_reward(is_correct)
            self.done = True
            
        else:
            return self._get_observation(), -1.0, False, {"error": "Unknown command"}
            
        # Update curriculum if used
        if self.use_curriculum and self.done:
            self.curriculum.update_difficulty(
                reward_info.get("total", 0) > 0
            )
            
        # Increment step counter
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True
            
        return self._get_observation(), reward_info["total"], self.done, reward_info
        
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        obs = {
            "current_query": self.current_query,
            "retrieved_items": self.retrieved_items,
            "memory_size": len(self.memory_buffer),
            "steps_remaining": self.max_steps - self.current_step,
            "memory_buffer": self.memory_buffer.copy(),
            "buffer_size": self.buffer_size,
            "available_space": self.buffer_size - len(self.memory_buffer)
        }
        
        if self.use_curriculum:
            obs["current_difficulty"] = self.curriculum.current_difficulty
            
        return obs
        
    def _generate_task(self) -> TaskConfig:
        """Generate a new memory task.
        
        Returns:
            TaskConfig: New task configuration
        """
        difficulty = self.curriculum.current_difficulty if self.use_curriculum else 0.5
        
        # Scale task complexity with difficulty
        num_required_items = max(1, int(2 + difficulty * 4))
        info_complexity = 0.2 + difficulty * 0.8
        
        task_types = [
            "factual_recall",
            "relational_query",
            "temporal_sequence",
            "categorical_grouping"
        ]
        
        selected_type = task_types[min(int(difficulty * len(task_types)), len(task_types) - 1)]
        
        return TaskConfig(
            difficulty=difficulty,
            task_type="memory_buffer",
            parameters={
                "type": selected_type,
                "required_items": num_required_items,
                "complexity": info_complexity,
                "buffer_size": self.buffer_size
            }
        )
    
    def _is_valid_action(self, action: str) -> bool:
        """Check if the action is valid.
        
        Args:
            action (str): Action to validate
            
        Returns:
            bool: Whether action is valid
        """
        try:
            command, content = action.split(" ", 1)
            command = command.upper()
            
            if command not in ["STORE", "RETRIEVE", "ANSWER"]:
                return False
                
            if not content.strip():
                return False
                
            if command == "STORE" and len(self.memory_buffer) >= self.buffer_size:
                return False
                
            return True
            
        except ValueError:
            return False
    
    def _compute_reward(self, action: str) -> float:
        """Compute reward for the current action.
        
        Args:
            action (str): Action taken
            
        Returns:
            float: Computed reward
        """
        try:
            command, content = action.split(" ", 1)
            command = command.upper()
            
            base_reward = 0.0
            
            if command == "STORE":
                # Small cost for storage to encourage efficiency
                base_reward = -0.1
                
                # Bonus for storing relevant information
                if self.current_query and any(term in content.lower() for term in self.current_query.lower().split()):
                    base_reward += 0.2
                    
            elif command == "RETRIEVE":
                self.total_retrievals += 1
                
                # Check if retrieval was relevant
                retrieved_relevant = False
                for item in self.memory_buffer:
                    if any(term in item.lower() for term in content.lower().split()):
                        retrieved_relevant = True
                        self.successful_retrievals += 1
                        break
                        
                base_reward = 0.2 if retrieved_relevant else -0.1
                
            elif command == "ANSWER":
                # Major reward for correct answer
                if self.target_response and content.lower() in self.target_response.lower():
                    base_reward = 1.0
                    # Bonus for efficiency
                    retrieval_efficiency = self.successful_retrievals / max(1, self.total_retrievals)
                    base_reward += 0.5 * retrieval_efficiency
                else:
                    base_reward = -0.2
            
            # Scale reward by task difficulty
            difficulty = self.current_task.difficulty if self.current_task else 0.5
            final_reward = base_reward * (1.0 + difficulty)
            
            return final_reward
            
        except ValueError:
            return -0.5
        
    def _store_memory(self, content: str) -> Dict[str, float]:
        """Store information in memory buffer."""
        # Extract topic (simple implementation)
        words = content.split()
        topic = words[0].lower() if words else "general"
        
        # Add to memory buffer
        memory_item = {
            "topic": topic,
            "content": content,
            "timestamp": self.current_step
        }
        
        if len(self.memory_buffer) >= self.buffer_size:
            # Remove oldest item
            self.memory_buffer.pop(0)
            
        self.memory_buffer.append(memory_item)
        
        # Compute reward
        state = {
            "task_completed": False,
            "task_difficulty": self.current_task.difficulty if self.current_task else 0.5,
            "memory_quality": len(content.split()) / 50.0  # Simple quality metric
        }
        
        return self.reward_manager.compute_reward(state)
        
    def _retrieve_memory(self, query: str) -> Dict[str, float]:
        """Retrieve information from memory buffer."""
        query = query.lower()
        relevant_items = []
        
        # Simple keyword matching (in practice, use more sophisticated retrieval)
        for idx, item in enumerate(self.memory_buffer):
            if query in item["content"].lower() or query in item["topic"]:
                relevant_items.append(item)
                
        self.retrieved_items = relevant_items
        
        # Compute reward based on relevance
        state = {
            "task_completed": False,
            "task_difficulty": self.current_task.difficulty if self.current_task else 0.5,
            "retrieval_success": len(relevant_items) > 0
        }
        
        return self.reward_manager.compute_reward(state)
        
    def _evaluate_answer(self, answer: str) -> bool:
        """Evaluate whether an answer is correct."""
        # Simple evaluation based on retrieved items
        if not self.retrieved_items:
            return False
            
        # Check if answer contains information from retrieved items
        return any(
            item["content"].lower() in answer.lower()
            for item in self.retrieved_items
        )
        
    def _compute_answer_reward(self, is_correct: bool) -> Dict[str, float]:
        """Compute reward for an answer."""
        state = {
            "task_completed": is_correct,
            "task_difficulty": self.current_task.difficulty if self.current_task else 0.5,
            "retrieval_used": len(self.retrieved_items) > 0
        }
        
        return self.reward_manager.compute_reward(state) 