from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
import random
import yaml
import os
from pathlib import Path

@dataclass
class TaskConfig:
    """Configuration for a task."""
    difficulty: float  # 0.0 to 1.0
    task_type: str
    parameters: Dict[str, Any]

class CurriculumGenerator:
    """Generates and manages curriculum of tasks with dynamic difficulty adjustment."""
    
    def __init__(self, 
                 initial_difficulty: float = 0.1,
                 difficulty_step: float = 0.05,
                 success_threshold: float = 0.7,
                 window_size: int = 10,
                 task_templates_path: Optional[str] = None):
        """Initialize curriculum generator.
        
        Args:
            initial_difficulty (float): Starting difficulty (0.0 to 1.0)
            difficulty_step (float): How much to adjust difficulty
            success_threshold (float): Success rate threshold for increasing difficulty
            window_size (int): Number of episodes to consider for adjustment
            task_templates_path (Optional[str]): Path to YAML file with task templates
        """
        self.current_difficulty = initial_difficulty
        self.difficulty_step = difficulty_step
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.recent_successes = []
        
        # Load task templates
        self.task_templates = {
            "logic_puzzle": self._generate_logic_puzzle,
            "dialogue": self._generate_dialogue_task,
            "meta_reasoning": self._generate_meta_task,
            "memory": self._generate_memory_task
        }
        
        # Load custom templates if provided
        if task_templates_path:
            self._load_task_templates(task_templates_path)
            
    def _load_task_templates(self, path: str):
        """Load task templates from YAML file."""
        if not os.path.exists(path):
            return
            
        with open(path, 'r') as f:
            templates = yaml.safe_load(f)
            
        # Update templates with custom ones
        for task_type, template in templates.items():
            if "generator" in template:
                # Custom generator function provided
                generator_path = template["generator"]
                module_path, func_name = generator_path.rsplit(".", 1)
                module = __import__(module_path, fromlist=[func_name])
                self.task_templates[task_type] = getattr(module, func_name)
            else:
                # Static template parameters
                self.task_templates[task_type] = lambda d, t=template: self._generate_from_template(d, t)
                
    def _generate_from_template(self, difficulty: float, template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate task parameters from a static template."""
        params = {}
        for key, value in template.items():
            if isinstance(value, list):
                # Select more items as difficulty increases
                num_items = max(1, int(difficulty * len(value)))
                params[key] = random.sample(value, k=num_items)
            elif isinstance(value, dict):
                # Recursively process nested templates
                params[key] = self._generate_from_template(difficulty, value)
            else:
                params[key] = value
        return params
        
    def generate_task(self, task_type: Optional[str] = None) -> TaskConfig:
        """Generate a new task with current difficulty level.
        
        Args:
            task_type (Optional[str]): Specific task type to generate
            
        Returns:
            TaskConfig: Configuration for the task
        """
        if task_type is None:
            task_type = random.choice(list(self.task_templates.keys()))
            
        generator = self.task_templates[task_type]
        parameters = generator(self.current_difficulty)
        
        return TaskConfig(
            difficulty=self.current_difficulty,
            task_type=task_type,
            parameters=parameters
        )
        
    def update_difficulty(self, success: bool):
        """Update difficulty based on recent performance.
        
        Args:
            success (bool): Whether the last task was completed successfully
        """
        self.recent_successes.append(float(success))
        if len(self.recent_successes) > self.window_size:
            self.recent_successes.pop(0)
            
        # Calculate success rate over window
        success_rate = np.mean(self.recent_successes)
        
        # Adjust difficulty
        if success_rate >= self.success_threshold:
            self.current_difficulty = min(1.0, self.current_difficulty + self.difficulty_step)
        elif success_rate < self.success_threshold - 0.2:  # Add some hysteresis
            self.current_difficulty = max(0.0, self.current_difficulty - self.difficulty_step)
            
    def _generate_logic_puzzle(self, difficulty: float) -> Dict[str, Any]:
        """Generate logic puzzle parameters based on difficulty.
        
        Args:
            difficulty (float): Current difficulty level
            
        Returns:
            Dict[str, Any]: Puzzle parameters
        """
        # Number of steps increases with difficulty
        num_steps = int(2 + difficulty * 8)  # 2 to 10 steps
        
        templates = [
            "If A then B",
            "Either A or B, but not both",
            "Neither A nor B",
            "If A then B, and if B then C",
            "A if and only if B",
            "All of A, B, and C",
            "At least one of A, B, or C",
            "If A then not B",
            "A implies B or C",
            "If A and B then C"
        ]
        
        # Select more complex templates as difficulty increases
        num_templates = max(1, int(difficulty * len(templates)))
        selected_templates = random.sample(templates[:num_templates], 
                                        k=min(num_steps, num_templates))
        
        # Generate variables with meaningful names at higher difficulties
        if difficulty > 0.5:
            variables = [
                "Raining", "Wet", "Umbrella", "Inside", "Happy",
                "Studying", "Pass", "Graduate", "Job", "Success"
            ][:num_steps]
        else:
            variables = ["A", "B", "C", "D", "E"][:num_steps]
        
        return {
            "num_steps": num_steps,
            "templates": selected_templates,
            "variables": variables,
            "time_limit": int(30 + (1.0 - difficulty) * 90)  # More time for easier puzzles
        }
        
    def _generate_dialogue_task(self, difficulty: float) -> Dict[str, Any]:
        """Generate dialogue task parameters based on difficulty.
        
        Args:
            difficulty (float): Current difficulty level
            
        Returns:
            Dict[str, Any]: Dialogue parameters
        """
        # More complex dialogue scenarios with higher difficulty
        num_turns = int(3 + difficulty * 7)  # 3 to 10 turns
        
        scenarios = [
            {
                "type": "simple_question_answer",
                "context": "A student asking about homework."
            },
            {
                "type": "negotiation",
                "context": "Negotiating a project deadline."
            },
            {
                "type": "debate",
                "context": "Discussing pros and cons of remote work."
            },
            {
                "type": "emotional_support",
                "context": "Supporting someone who failed an exam."
            },
            {
                "type": "technical_support",
                "context": "Helping with software installation issues."
            }
        ]
        
        # More complex scenarios at higher difficulties
        available_scenarios = scenarios[:max(1, int(difficulty * len(scenarios)))]
        scenario = random.choice(available_scenarios)
        
        return {
            "scenario": scenario["type"],
            "context": scenario["context"],
            "num_turns": num_turns,
            "required_elements": [
                "politeness",
                "coherence",
                *(["emotional_awareness"] if difficulty > 0.3 else []),
                *(["conflict_resolution"] if difficulty > 0.6 else []),
                *(["cultural_sensitivity"] if difficulty > 0.8 else [])
            ]
        }
        
    def _generate_meta_task(self, difficulty: float) -> Dict[str, Any]:
        """Generate meta-reasoning task parameters based on difficulty.
        
        Args:
            difficulty (float): Current difficulty level
            
        Returns:
            Dict[str, Any]: Meta-reasoning parameters
        """
        # More complex meta-reasoning requirements with higher difficulty
        num_reflections = int(1 + difficulty * 4)  # 1 to 5 reflection steps
        
        reflection_types = [
            {
                "type": "simple_verification",
                "prompt": "Verify your basic assumptions."
            },
            {
                "type": "assumption_checking",
                "prompt": "What assumptions are you making?"
            },
            {
                "type": "alternative_approaches",
                "prompt": "What other approaches could work?"
            },
            {
                "type": "bias_detection",
                "prompt": "What biases might affect your thinking?"
            },
            {
                "type": "uncertainty_quantification",
                "prompt": "How certain are you about this?"
            }
        ]
        
        # Select more reflection types as difficulty increases
        num_types = max(1, int(difficulty * len(reflection_types)))
        selected_types = random.sample(reflection_types[:num_types], 
                                     k=min(num_reflections, num_types))
        
        return {
            "num_reflections": num_reflections,
            "reflection_types": [t["type"] for t in selected_types],
            "prompts": [t["prompt"] for t in selected_types],
            "required_justification": difficulty > 0.5,
            "uncertainty_tracking": difficulty > 0.7
        }
        
    def _generate_memory_task(self, difficulty: float) -> Dict[str, Any]:
        """Generate memory task parameters based on difficulty."""
        # More complex memory tasks with higher difficulty
        num_items = int(2 + difficulty * 8)  # 2 to 10 items to remember
        
        task_types = [
            "factual_recall",  # Simple fact storage and retrieval
            "relational_memory",  # Store and recall relationships
            "temporal_sequence",  # Remember order of events
            "categorical_organization",  # Organize information by category
            "abstract_concepts"  # Store and retrieve abstract ideas
        ]
        
        # Select task type based on difficulty
        available_types = task_types[:max(1, int(difficulty * len(task_types)))]
        task_type = random.choice(available_types)
        
        # Generate topics based on task type
        topics = {
            "factual_recall": ["history", "science", "geography"],
            "relational_memory": ["family_tree", "cause_effect", "dependencies"],
            "temporal_sequence": ["story_events", "instructions", "timeline"],
            "categorical_organization": ["animals", "plants", "professions"],
            "abstract_concepts": ["theories", "principles", "philosophies"]
        }
        
        selected_topics = random.sample(
            topics[task_type],
            k=min(num_items, len(topics[task_type]))
        )
        
        return {
            "task_type": task_type,
            "num_items": num_items,
            "topics": selected_topics,
            "require_ordering": task_type == "temporal_sequence",
            "require_categorization": task_type == "categorical_organization",
            "abstract_reasoning": task_type == "abstract_concepts"
        } 