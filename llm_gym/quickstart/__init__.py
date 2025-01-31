"""Quickstart module for LLM GYM with Qwen2.5-3B-Instruct model."""

import torch
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenAgent:
    """Agent using Qwen model for LLM GYM environments."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        """Initialize the Qwen-based agent.
        
        Args:
            model_name: Name of the Qwen model to use
        """
        print(f"Loading {model_name} on cuda...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        print("Model loaded successfully!")
        
        # Environment-specific prompt templates
        self.prompt_templates = {
            "LogicPuzzleEnv": """You are solving a logic puzzle. Given the premises, you need to determine logical conclusions.

Current Premises: {premises}
Attempts Made: {num_attempts}
Steps Remaining: {steps_remaining}
Difficulty: {current_difficulty}

You can use these commands:
- SOLVE <conclusion>: Submit a logical conclusion
- QUERY <question>: Ask about relationships between variables

Based on the premises, what action would you take? Respond with just the command and your answer.

Your response should be in the format:
SOLVE <your logical conclusion>
or
QUERY <your question>""",
            
            "DialogueEnv": """You are engaging in a dialogue task.

Context: {context}
Required Elements: {required_elements}
Current Turn: {current_turn}
Goals Achieved: {goals_achieved}

Respond appropriately to continue the conversation while incorporating required elements.

Your response:""",
            
            "MemoryBufferEnv": """You are managing a memory buffer.

Current Query: {query}
Buffer Status: {buffer_status}
Retrieved Items: {retrieved_items}

Available Commands:
- STORE <content>: Store information
- RETRIEVE <query>: Search memory
- ANSWER <response>: Provide final answer

Your action:"""
        }
    
    def generate_response(self, 
                         prompt: str, 
                         max_length: int = 100,
                         temperature: float = 0.7) -> str:
        """Generate a response using the Qwen model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum response length
            temperature: Sampling temperature
            
        Returns:
            str: Generated response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the actual response after the prompt
        response = response[len(prompt):].strip()
        return response
    
    def act(self, observation: Dict[str, Any]) -> str:
        """Generate an action based on the current observation.
        
        Args:
            observation: Current environment observation
            
        Returns:
            str: Selected action
        """
        # Determine environment type from observation structure
        env_type = self._detect_env_type(observation)
        
        if env_type not in self.prompt_templates:
            raise ValueError(f"Unknown environment type: {env_type}")
        
        # Format prompt using environment-specific template
        prompt = self.prompt_templates[env_type].format(**observation)
        
        # Generate and format response
        response = self.generate_response(prompt)
        
        # Ensure response follows environment-specific format
        formatted_response = self._format_response(response, env_type)
        
        return formatted_response
    
    def _detect_env_type(self, observation: Dict[str, Any]) -> str:
        """Detect environment type from observation structure.
        
        Args:
            observation: Current observation
            
        Returns:
            str: Detected environment type
        """
        if "premises" in observation:
            return "LogicPuzzleEnv"
        elif "context" in observation:
            return "DialogueEnv"
        elif "buffer_status" in observation:
            return "MemoryBufferEnv"
        else:
            raise ValueError("Unable to detect environment type from observation")
    
    def _format_response(self, response: str, env_type: str) -> str:
        """Format response according to environment requirements.
        
        Args:
            response: Raw model response
            env_type: Type of environment
            
        Returns:
            str: Formatted response
        """
        response = response.strip()
        
        if env_type == "LogicPuzzleEnv":
            # Ensure response starts with SOLVE or QUERY
            if not (response.startswith("SOLVE") or response.startswith("QUERY")):
                if "therefore" in response.lower() or "conclusion" in response.lower():
                    response = "SOLVE " + response
                else:
                    response = "QUERY " + response
        
        return response 