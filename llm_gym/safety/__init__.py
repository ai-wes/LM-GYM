"""Safety and ethics module for LLM GYM."""

from typing import List, Dict, Any, Tuple, Callable
from abc import ABC, abstractmethod

class SafetyPolicy(ABC):
    """Abstract base class for safety policies."""
    
    @abstractmethod
    def check(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """Check text against policy.
        
        Args:
            text (str): Text to check
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (is_safe, details)
        """
        pass

class ContentSafetyPolicy(SafetyPolicy):
    """Checks content for harmful language."""
    
    def __init__(self, harmful_terms: List[str] = None):
        self.harmful_terms = harmful_terms or {
            "harm", "hate", "violence", "discriminate",
            "threat", "abuse", "kill", "attack"
        }
        
    def check(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """Check text for harmful content."""
        text = text.lower()
        found_terms = [term for term in self.harmful_terms 
                      if term in text]
        return (
            len(found_terms) == 0,
            {
                "type": "content_safety",
                "found_terms": found_terms,
                "severity": len(found_terms)
            }
        )

class FairnessPolicy(SafetyPolicy):
    """Checks for bias and fairness issues."""
    
    def __init__(self, sensitive_terms: List[str] = None):
        self.sensitive_terms = sensitive_terms or {
            "race", "gender", "religion", "ethnicity",
            "nationality", "disability", "age", "orientation"
        }
        
    def check(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """Check text for fairness issues."""
        text = text.lower()
        mentioned = [term for term in self.sensitive_terms 
                    if term in text]
        return (
            len(mentioned) == 0,
            {
                "type": "fairness",
                "mentioned_topics": mentioned,
                "severity": len(mentioned)
            }
        )

class PrivacyPolicy(SafetyPolicy):
    """Checks for potential privacy violations."""
    
    def __init__(self, pii_patterns: List[str] = None):
        self.pii_patterns = pii_patterns or {
            "email", "phone", "address", "ssn", "password",
            "credit card", "bank account", "social security"
        }
        
    def check(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """Check text for privacy issues."""
        text = text.lower()
        found_pii = [pattern for pattern in self.pii_patterns 
                    if pattern in text]
        return (
            len(found_pii) == 0,
            {
                "type": "privacy",
                "found_pii": found_pii,
                "severity": len(found_pii)
            }
        )

class SafetyManager:
    """Ensures agent behavior aligns with ethical guidelines."""
    
    def __init__(self, policies: List[str] = None):
        """Initialize safety manager.
        
        Args:
            policies (List[str]): List of policy names to enable
        """
        self.policies = self._load_policies(policies or ["content_safety", "fairness"])
        
    def _load_policies(self, policy_names: List[str]) -> List[SafetyPolicy]:
        """Load specified safety policies.
        
        Args:
            policy_names (List[str]): Names of policies to load
            
        Returns:
            List[SafetyPolicy]: List of policy instances
        """
        policy_map = {
            "content_safety": ContentSafetyPolicy,
            "fairness": FairnessPolicy,
            "privacy": PrivacyPolicy
        }
        
        return [policy_map[name]() for name in policy_names 
                if name in policy_map]
        
    def check_action(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        """Check an action against all safety policies.
        
        Args:
            action (str): Action to check
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (is_safe, details)
        """
        violations = []
        for policy in self.policies:
            safe, result = policy.check(action)
            if not safe:
                violations.append(result)
                
        return (
            len(violations) == 0,
            {
                "violations": violations,
                "total_violations": len(violations),
                "max_severity": max([v["severity"] for v in violations], default=0)
            }
        )
        
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get statistics about policy violations."""
        return {
            "num_policies": len(self.policies),
            "policy_types": [p.__class__.__name__ for p in self.policies]
        } 