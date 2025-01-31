"""LLM GYM - A framework for training and evaluating LLM cognitive strategies."""

from llm_gym.envs.base import BaseEnv
from llm_gym.envs.meta_reasoning import MetaReasoningEnv
from llm_gym.envs.logic_puzzle import LogicPuzzleEnv
from llm_gym.envs.dialogue import DialogueEnv
from llm_gym.envs.memory_buffer import MemoryBufferEnv

from llm_gym.safety import (
    SafetyManager,
    SafetyPolicy,
    ContentSafetyPolicy,
    FairnessPolicy,
    PrivacyPolicy
)

from llm_gym.curriculum.enhanced import (
    EnhancedCurriculum,
    TaskMetrics,
    PerformanceTracker,
    AdaptiveDifficulty
)

from llm_gym.monitoring.visualization import (
    MetricsLogger,
    PerformanceVisualizer,
    TrainingMetrics
)

from llm_gym.rewards.adaptive import (
    AdaptiveRewardManager,
    RewardComponent,
    RewardFunction,
    TaskCompletionReward,
    EfficiencyReward,
    SafetyReward,
    NoveltyReward
)

from llm_gym.distributed import (
    RemoteEnvironment,
    DistributedTrainer
)

__version__ = "0.2.0"

__all__ = [
    # Base Environment
    "BaseEnv",
    
    # Environment Types
    "MetaReasoningEnv",
    "LogicPuzzleEnv",
    "DialogueEnv",
    "MemoryBufferEnv",
    
    # Safety Module
    "SafetyManager",
    "SafetyPolicy",
    "ContentSafetyPolicy",
    "FairnessPolicy",
    "PrivacyPolicy",
    
    # Enhanced Curriculum
    "EnhancedCurriculum",
    "TaskMetrics",
    "PerformanceTracker",
    "AdaptiveDifficulty",
    
    # Monitoring and Visualization
    "MetricsLogger",
    "PerformanceVisualizer",
    "TrainingMetrics",
    
    # Adaptive Rewards
    "AdaptiveRewardManager",
    "RewardComponent",
    "RewardFunction",
    "TaskCompletionReward",
    "EfficiencyReward",
    "SafetyReward",
    "NoveltyReward",
    
    # Distributed Training
    "RemoteEnvironment",
    "DistributedTrainer"
] 